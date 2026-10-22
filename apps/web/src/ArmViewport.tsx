import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useI18n } from "./i18n";

interface Props {
  joints: number[];
  target: number[];
  status: string;
}

function createLink(length: number, radius: number, color: number) {
  const group = new THREE.Group();
  const mesh = new THREE.Mesh(
    new THREE.CapsuleGeometry(radius, length - radius * 2, 8, 16),
    new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.25 }),
  );
  mesh.rotation.z = -Math.PI / 2;
  mesh.position.x = length / 2;
  mesh.castShadow = true;
  group.add(mesh);
  return group;
}

export function ArmViewport({ joints, target, status }: Props) {
  const { t } = useI18n();
  const hostRef = useRef<HTMLDivElement>(null);
  const jointsRef = useRef(joints);
  const targetRef = useRef(target);

  useEffect(() => { jointsRef.current = joints; }, [joints]);
  useEffect(() => { targetRef.current = target; }, [target]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe8ebe4);
    scene.fog = new THREE.Fog(0xe8ebe4, 6, 12);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 30);
    camera.position.set(4.8, 3.3, 5.3);
    camera.lookAt(0.6, 1.1, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x53635d, 2.3));
    const keyLight = new THREE.DirectionalLight(0xfff4dc, 3.2);
    keyLight.position.set(3, 6, 4);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    scene.add(keyLight);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(14, 14),
      new THREE.MeshStandardMaterial({ color: 0xdde2da, roughness: 0.92 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    const grid = new THREE.GridHelper(14, 28, 0x99a69f, 0xc6cec8);
    grid.position.y = 0.003;
    scene.add(grid);

    const baseMaterial = new THREE.MeshStandardMaterial({ color: 0x263633, roughness: 0.38, metalness: 0.4 });
    const jointMaterial = new THREE.MeshStandardMaterial({ color: 0xf5b944, roughness: 0.35, metalness: 0.15 });
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.43, 0.5, 0.35, 32), baseMaterial);
    base.position.y = 0.18;
    base.castShadow = true;
    scene.add(base);

    const root = new THREE.Group();
    root.position.y = 0.42;
    scene.add(root);
    const shoulder = new THREE.Group();
    const elbow = new THREE.Group();
    const wrist = new THREE.Group();
    root.add(shoulder);
    shoulder.add(createLink(1.45, 0.16, 0x263633));
    elbow.position.x = 1.45;
    shoulder.add(elbow);
    elbow.add(createLink(1.2, 0.14, 0x36514b));
    wrist.position.x = 1.2;
    elbow.add(wrist);
    wrist.add(createLink(0.82, 0.11, 0x263633));
    [shoulder, elbow, wrist].forEach((joint) => {
      const housing = new THREE.Mesh(new THREE.SphereGeometry(0.21, 24, 16), jointMaterial);
      housing.castShadow = true;
      joint.add(housing);
    });
    const gripper = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.34, 0.34), new THREE.MeshStandardMaterial({ color: 0xe05a3f }));
    gripper.position.x = 0.88;
    gripper.castShadow = true;
    wrist.add(gripper);

    const targetGroup = new THREE.Group();
    const targetRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.22, 0.035, 10, 40),
      new THREE.MeshStandardMaterial({ color: 0xe05a3f, emissive: 0x5b140b }),
    );
    targetRing.rotation.x = Math.PI / 2;
    targetGroup.add(targetRing);
    targetGroup.add(new THREE.Mesh(new THREE.SphereGeometry(0.07, 18, 12), new THREE.MeshBasicMaterial({ color: 0xe05a3f })));
    scene.add(targetGroup);

    let frame = 0;
    const resize = () => {
      const width = Math.max(host.clientWidth, 1);
      const height = Math.max(host.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const render = () => {
      const current = jointsRef.current;
      root.rotation.y = current[0] ?? 0;
      shoulder.rotation.z = (current[1] ?? 0) - 0.55;
      elbow.rotation.z = (current[2] ?? 0) + 0.72;
      wrist.rotation.z = -0.18;
      const currentTarget = targetRef.current;
      targetGroup.position.set(
        0.8 + (currentTarget[0] ?? 0) * 1.7,
        0.3 + Math.max(0, currentTarget[2] ?? 0) * 1.8,
        (currentTarget[1] ?? 0) * 1.7,
      );
      targetGroup.rotation.y += 0.008;
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };
    render();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div className="viewport">
      <div ref={hostRef} className="simulation-canvas" role="img" aria-label={t("Mock 三关节机械臂三维仿真视图", "Mock 3D three-joint robot simulation")} />
      <div className="viewport-status"><span className={`status-dot ${status}`} />{status}</div>
      <div className="viewport-label">{t("Mock 3D · 几何近似", "MOCK 3D · GEOMETRIC APPROXIMATION")}</div>
    </div>
  );
}
