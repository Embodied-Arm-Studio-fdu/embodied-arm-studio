import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import URDFLoader, { type URDFRobot } from "urdf-loader";
import type { SceneObject, SceneDocument } from "./api";
import { S1_ARM_JOINTS } from "./s1Robot";
import { useI18n } from "./i18n";

interface Props {
  scene: SceneDocument;
  selectedObjectId: string;
  onSelect: (objectId: string) => void;
  jointPositions: number[];
}

function makeGeometry(item: SceneObject) {
  if (item.type === "sphere") return new THREE.SphereGeometry(0.5, 24, 16);
  if (item.type === "cylinder") return new THREE.CylinderGeometry(0.5, 0.5, 1, 24);
  if (item.type === "wall") return new THREE.BoxGeometry(1.8, 1, 0.12);
  if (item.type === "target") return new THREE.CylinderGeometry(0.5, 0.5, 0.04, 32);
  return new THREE.BoxGeometry(1, 1, 1);
}

function isS1Scene(scene: SceneDocument): boolean {
  return scene.capability !== "mock" && scene.robot.dof === S1_ARM_JOINTS.length;
}

function disposeObject(object: THREE.Object3D) {
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.geometry.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((material) => material.dispose());
  });
}

function setS1JointValues(robot: URDFRobot, positions: number[]) {
  S1_ARM_JOINTS.forEach((joint, index) => {
    robot.joints[joint.name]?.setJointValue(positions[index] ?? 0);
  });
}

export function SceneEditor({ scene, selectedObjectId, onSelect, jointPositions }: Props) {
  const { t } = useI18n();
  const s1Scene = isS1Scene(scene);
  const hostRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef(scene);
  const rebuildRef = useRef<(() => void) | null>(null);
  const selectedRef = useRef(selectedObjectId);
  const onSelectRef = useRef(onSelect);
  const updateArmRef = useRef<((positions: number[]) => void) | null>(null);
  const [modelStatus, setModelStatus] = useState<"loading" | "ready" | "error" | null>(null);
  useEffect(() => {
    sceneRef.current = scene;
    rebuildRef.current?.();
  }, [scene]);
  useEffect(() => { selectedRef.current = selectedObjectId; }, [selectedObjectId]);
  useEffect(() => { onSelectRef.current = onSelect; }, [onSelect]);
  useEffect(() => { updateArmRef.current?.(jointPositions); }, [jointPositions]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let disposed = false;
    const world = new THREE.Scene();
    world.background = new THREE.Color(0xe8ebe4);
    world.fog = new THREE.Fog(0xe8ebe4, 8, 18);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 30);
    camera.position.set(1.35, 1.05, 1.55);
    camera.lookAt(0.15, 0.35, 0);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);
    world.add(new THREE.HemisphereLight(0xffffff, 0x53635d, 2.1));
    const keyLight = new THREE.DirectionalLight(0xfff4dc, 3);
    keyLight.position.set(3, 6, 4);
    keyLight.castShadow = true;
    world.add(keyLight);
    const grid = new THREE.GridHelper(2.4, 24, 0x87968e, 0xc6cec8);
    grid.position.y = 0.01;
    world.add(grid);
    if (s1Scene) {
      setModelStatus("loading");
      let assetLoadFailed = false;
      const loadingManager = new THREE.LoadingManager();
      loadingManager.onError = () => {
        assetLoadFailed = true;
        if (!disposed) setModelStatus("error");
      };
      loadingManager.onLoad = () => {
        if (!disposed && !assetLoadFailed) setModelStatus("ready");
      };
      const loader = new URDFLoader(loadingManager);
      loader.parseCollision = false;
      loader.load(
        "/assets/s1/yhrg_s1_description/urdf/yhrg_s1_v2.urdf",
        (robot) => {
          if (disposed) {
            disposeObject(robot);
            return;
          }
          robot.rotation.x = -Math.PI / 2;
          robot.position.set(0, 0.12, 0);
          robot.traverse((object) => {
            if (object instanceof THREE.Mesh) {
              object.castShadow = true;
              object.receiveShadow = true;
            }
          });
          world.add(robot);
          updateArmRef.current = (positions) => setS1JointValues(robot, positions);
          updateArmRef.current(jointPositions);
        },
        undefined,
        () => { if (!disposed) setModelStatus("error"); },
      );
    } else {
      setModelStatus(null);
    }
    const objectGroups = new Map<string, THREE.Group>();
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const rebuild = () => {
      objectGroups.forEach((group) => world.remove(group));
      objectGroups.clear();
      sceneRef.current.objects.filter((item) => item.type !== "plane").forEach((item) => {
        const group = new THREE.Group();
        group.name = item.object_id;
        group.position.set(item.position[0], item.position[1], item.position[2]);
        group.rotation.set(item.rotation[0], item.rotation[1], item.rotation[2]);
        group.scale.set(item.scale[0], item.scale[1], item.scale[2]);
        const material = new THREE.MeshStandardMaterial({ color: item.color, roughness: 0.55, metalness: item.type === "target" ? 0.05 : 0.18, transparent: item.type === "target", opacity: item.type === "target" ? 0.7 : 1 });
        const mesh = new THREE.Mesh(makeGeometry(item), material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
        world.add(group);
        objectGroups.set(item.object_id, group);
      });
    };
    rebuildRef.current = rebuild;
    rebuild();
    const handleClick = (event: PointerEvent) => {
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects([...objectGroups.values()], true)[0];
      if (hit) onSelectRef.current(hit.object.parent?.name || hit.object.name);
    };
    renderer.domElement.addEventListener("pointerdown", handleClick);
    let frame = 0;
    const resize = () => { const width = Math.max(host.clientWidth, 1); const height = Math.max(host.clientHeight, 1); renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix(); };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    const render = () => {
      objectGroups.forEach((group, objectId) => { group.children.forEach((child) => { if (child instanceof THREE.Mesh) child.material.emissive?.setHex(objectId === selectedRef.current ? 0x3b7568 : 0x000000); }); });
      renderer.render(world, camera);
      frame = requestAnimationFrame(render);
    };
    render();
    return () => { disposed = true; rebuildRef.current = null; updateArmRef.current = null; cancelAnimationFrame(frame); observer.disconnect(); renderer.domElement.removeEventListener("pointerdown", handleClick); disposeObject(world); renderer.dispose(); renderer.domElement.remove(); };
  }, [s1Scene]);

  return <div className="scene-editor-canvas"><div ref={hostRef} className="scene-editor-webgl" role="img" aria-label={s1Scene ? t("S1 六自由度机械臂真实网格与可编辑实验场景预览", "S1 six-DoF robot mesh and editable experiment scene preview") : t("简化教学场景对象预览", "Simplified teaching scene preview")} /><span className="scene-editor-badge">{s1Scene ? t("S1 六轴真实网格预览", "S1 six-axis mesh preview") : t("入门简化场景预览", "Introductory simplified scene")} · {t("场景单位", "Scene units")} m</span><span className="scene-editor-help">{t("点击物体选择", "Select an object by clicking")} · {s1Scene ? t("真实网格 · 控制与碰撞验证关闭", "Production mesh · control and collision validation disabled") : t("当前场景未接入 S1 资产", "S1 assets are not connected to this scene")}</span>{s1Scene && <span className={`scene-editor-model-status ${modelStatus ?? "loading"}`} role="status">{modelStatus === "ready" ? t("S1 网格已加载", "S1 mesh loaded") : modelStatus === "error" ? t("S1 网格不可用", "S1 mesh unavailable") : t("正在加载 S1 网格", "Loading S1 mesh")}</span>}</div>;
}
