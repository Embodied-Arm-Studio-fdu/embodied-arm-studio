import { useLearningText } from "./learningTranslations";
import { useMemo, useState, type KeyboardEvent } from "react";

type ArmPart = "base" | "shoulder" | "upper-arm" | "elbow" | "forearm" | "wrist" | "gripper" | "tcp" | "actuator" | "sensor";

const ARM_PARTS: Record<ArmPart, { label: string; description: string }> = {
  base: { label: "基座 Base", description: "固定机器人并承受整条串联关节链的载荷；第一轴通常提供基座回转。" },
  shoulder: { label: "肩部关节 Shoulder", description: "驱动上臂抬起或下降，对工作空间和负载能力影响显著。" },
  "upper-arm": { label: "上臂连杆 Upper arm", description: "连接肩部与肘部；长度决定机械臂主要可达范围的一部分。" },
  elbow: { label: "肘部关节 Elbow", description: "改变上臂与前臂夹角，使末端能够伸展或收拢。" },
  forearm: { label: "前臂连杆 Forearm", description: "连接肘部与腕部，把上游关节运动传递到腕部和工具。" },
  wrist: { label: "腕部关节 Wrist J4–J6", description: "三个相邻旋转轴主要调整末端姿态，而不是只改变位置。" },
  gripper: { label: "两指夹爪 Gripper", description: "末端执行器通过开合与物体交互；工具可按任务更换。" },
  tcp: { label: "工具中心点 TCP", description: "规划和标定常使用的工具参考点；其位置和姿态定义任务空间目标。" },
  actuator: { label: "编码器 / 执行器", description: "关节处的执行器产生运动，编码器测量关节位置并形成反馈。" },
  sensor: { label: "相机 / 力传感器", description: "腕部相机提供局部视觉，力传感器测量工具与环境之间的作用。" },
};

function keyboardActivate(event: KeyboardEvent<SVGGElement>, activate: () => void) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    activate();
  }
}

export function RobotArmAnatomyDiagram() {
  const lt = useLearningText();
  const [active, setActive] = useState<ArmPart>("shoulder");
  const bind = (part: ArmPart) => ({
    className: `robot-arm-part ${active === part ? "active" : ""}`,
    role: "button" as const,
    tabIndex: 0,
    "aria-label": `${lt(ARM_PARTS[part].label)}: ${lt(ARM_PARTS[part].description)}`,
    onMouseEnter: () => setActive(part),
    onFocus: () => setActive(part),
    onClick: () => setActive(part),
    onKeyDown: (event: KeyboardEvent<SVGGElement>) => keyboardActivate(event, () => setActive(part)),
  });

  return <figure className="learning-figure robotics-figure" data-learning-visual="arm-anatomy">
    <div className="robotics-svg-frame">
      <svg viewBox="0 0 900 500" role="img" aria-labelledby="arm-anatomy-title arm-anatomy-desc">
        <title id="arm-anatomy-title">{lt("经典六自由度串联机械臂构成图")}</title>
        <desc id="arm-anatomy-desc">{lt("从基座、肩部、上臂、肘部、前臂、三轴腕部到两指夹爪和工具中心点的教学矢量示意。各关节标有旋转轴、执行器和编码器位置，腕部标有可选相机与力传感器。")}</desc>
        <defs>
          <marker id="arm-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z"/></marker>
          <marker id="arm-arrow-small" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z"/></marker>
        </defs>

        <path className="robot-floor" d="M56 426 H830"/>
        <g {...bind("base")}><path className="arm-link" d="M112 405 H205 L190 365 H127 Z"/><ellipse className="arm-joint" cx="158" cy="365" rx="34" ry="18"/><text x="72" y="462">{lt("基座 Base · J1")}</text></g>
        <g {...bind("shoulder")}><circle className="arm-joint" cx="188" cy="332" r="29"/><text x="80" y="310">{lt("肩部关节 J2")}</text></g>
        <g {...bind("upper-arm")}><path className="arm-link" d="M204 315 L335 235"/><text x="222" y="250">{lt("上臂连杆")}</text></g>
        <g {...bind("elbow")}><circle className="arm-joint" cx="350" cy="226" r="27"/><text x="318" y="174">{lt("肘部 J3")}</text></g>
        <g {...bind("forearm")}><path className="arm-link" d="M370 218 L500 184"/><text x="405" y="168">{lt("前臂连杆")}</text></g>
        <g {...bind("wrist")}><circle className="arm-joint" cx="520" cy="179" r="23"/><circle className="arm-joint" cx="573" cy="178" r="20"/><circle className="arm-joint" cx="620" cy="178" r="18"/><path className="arm-link" d="M540 179 H602"/><text x="496" y="126">{lt("腕部 J4 / J5 / J6")}</text></g>
        <g {...bind("sensor")}><rect className="arm-sensor" x="574" y="212" width="35" height="24" rx="4"/><circle className="arm-camera" cx="591" cy="224" r="6"/><rect className="arm-force" x="638" y="165" width="18" height="26"/><text x="548" y="264">{lt("腕部相机（可选）")}</text><text x="626" y="218">{lt("力传感器")}</text></g>
        <g {...bind("gripper")}><path className="arm-link" d="M638 178 H692"/><path className="arm-gripper" d="M690 163 L730 147 M690 193 L730 209"/><text x="666" y="118">{lt("两指夹爪")}</text></g>
        <g {...bind("tcp")}><circle className="arm-tcp" cx="742" cy="178" r="6"/><path className="arm-tcp-cross" d="M728 178 H756 M742 164 V192"/><text x="758" y="184">TCP</text></g>

        <g {...bind("actuator")}><circle className="arm-encoder" cx="158" cy="365" r="8"/><circle className="arm-encoder" cx="188" cy="332" r="8"/><circle className="arm-encoder" cx="350" cy="226" r="8"/><circle className="arm-encoder" cx="520" cy="179" r="7"/><circle className="arm-encoder" cx="573" cy="178" r="7"/><circle className="arm-encoder" cx="620" cy="178" r="7"/><text x="92" y="82">{lt("橙色圆点：编码器 / 执行器所在位置")}</text></g>

        <g className="arm-axis-arcs" aria-label={lt("关节旋转轴")}>
          <path d="M126 352 A38 24 0 1 1 186 347" markerEnd="url(#arm-arrow-small)"/><path d="M160 303 A38 38 0 0 1 215 306" markerEnd="url(#arm-arrow-small)"/><path d="M321 201 A36 36 0 0 1 375 205" markerEnd="url(#arm-arrow-small)"/><path d="M495 151 A31 31 0 0 1 540 153" markerEnd="url(#arm-arrow-small)"/><path d="M552 151 A28 28 0 0 1 594 153" markerEnd="url(#arm-arrow-small)"/><path d="M601 151 A26 26 0 0 1 640 154" markerEnd="url(#arm-arrow-small)"/>
          <text x="90" y="342">{lt("旋转轴")}</text>
        </g>
        <path className="arm-callout" d="M608 223 C675 245 722 236 770 218" markerEnd="url(#arm-arrow)"/>
        <text x="716" y="252">{lt("末端传感")}</text>
      </svg>
    </div>
    <div className="robotics-part-picker" role="group" aria-label={lt("选择机械臂部件查看说明")}>{(Object.keys(ARM_PARTS) as ArmPart[]).map((part) => <button key={part} className={active === part ? "active" : ""} onClick={() => setActive(part)}>{lt(ARM_PARTS[part].label)}</button>)}</div>
    <div className="robotics-part-detail" aria-live="polite"><strong>{lt(ARM_PARTS[active].label)}</strong><p>{lt(ARM_PARTS[active].description)}</p></div>
    <figcaption><strong>{lt("经典六自由度串联机械臂构成示意")}</strong><span>{lt("帮助理解基本部件，并非S1机械臂精确结构图。")}</span><em>{lt("教学示意 · 本地 React SVG，无外部图片依赖")}</em></figcaption>
  </figure>;
}

function pointOnCircle(cx: number, cy: number, radius: number, degrees: number) {
  const angle = degrees * Math.PI / 180;
  return { x: cx + radius * Math.cos(angle), y: cy - radius * Math.sin(angle) };
}

function arcPath(cx: number, cy: number, radius: number, startDegrees: number, endDegrees: number) {
  const start = pointOnCircle(cx, cy, radius, startDegrees);
  const end = pointOnCircle(cx, cy, radius, endDegrees);
  const delta = endDegrees - startDegrees;
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${Math.abs(delta) > 180 ? 1 : 0} ${delta >= 0 ? 0 : 1} ${end.x} ${end.y}`;
}

export function PlanarKinematicsDiagram() {
  const lt = useLearningText();
  const l1 = .4;
  const l2 = .3;
  const [q1, setQ1] = useState(30);
  const [q2, setQ2] = useState(60);
  const values = useMemo(() => {
    const q1Rad = q1 * Math.PI / 180;
    const q12Rad = (q1 + q2) * Math.PI / 180;
    const x1 = l1 * Math.cos(q1Rad);
    const y1 = l1 * Math.sin(q1Rad);
    const x2 = l2 * Math.cos(q12Rad);
    const y2 = l2 * Math.sin(q12Rad);
    return { q1Rad, q12Rad, x1, y1, x2, y2, x: x1 + x2, y: y1 + y2 };
  }, [q1, q2]);
  const origin = { x: 125, y: 370 };
  const scale = 390;
  const elbow = { x: origin.x + values.x1 * scale, y: origin.y - values.y1 * scale };
  const tip = { x: origin.x + values.x * scale, y: origin.y - values.y * scale };
  const reset = () => { setQ1(30); setQ2(60); };

  return <figure className="learning-figure robotics-figure kinematics-figure" data-learning-visual="planar-kinematics">
    <div className="robotics-svg-frame">
      <svg viewBox="0 0 760 500" role="img" aria-labelledby="kinematics-title kinematics-desc">
        <title id="kinematics-title">{lt("二维二连杆正逆运动学交互矢量图")}</title>
        <desc id="kinematics-desc">{lt("显示基座原点、坐标轴、两根连杆、肘部和末端点、关节角、绝对方向以及水平和竖直分量。拖动关节角会根据正运动学公式实时更新。")}</desc>
        <defs><marker id="kin-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z"/></marker></defs>
        <path className="kin-axis" d={`M55 ${origin.y} H675`} markerEnd="url(#kin-arrow)"/><path className="kin-axis" d={`M${origin.x} 445 V48`} markerEnd="url(#kin-arrow)"/>
        <text x="684" y={origin.y + 6}>x</text><text x={origin.x - 18} y="44">y</text><text x={origin.x - 25} y={origin.y + 28}>O</text>

        <path className="kin-projection" d={`M${tip.x} ${tip.y} V${origin.y} M${tip.x} ${tip.y} H${origin.x}`}/>
        <path className="kin-component kin-component-one" d={`M${origin.x} ${origin.y} H${elbow.x} V${elbow.y}`}/>
        <path className="kin-component kin-component-two" d={`M${elbow.x} ${elbow.y} H${tip.x} V${tip.y}`}/>
        <path className="kin-link kin-link-one" d={`M${origin.x} ${origin.y} L${elbow.x} ${elbow.y}`}/>
        <path className="kin-link kin-link-two" d={`M${elbow.x} ${elbow.y} L${tip.x} ${tip.y}`}/>
        <circle className="kin-joint" cx={origin.x} cy={origin.y} r="9"/><circle className="kin-joint" cx={elbow.x} cy={elbow.y} r="9"/><circle className="kin-tip" cx={tip.x} cy={tip.y} r="9"/>

        <path className="kin-angle" d={arcPath(origin.x, origin.y, 62, 0, q1)} markerEnd="url(#kin-arrow)"/>
        <path className="kin-angle kin-angle-two" d={arcPath(elbow.x, elbow.y, 46, q1, q1 + q2)} markerEnd="url(#kin-arrow)"/>
        <text x={origin.x + 70} y={origin.y - 20}>q₁={q1}°</text>
        <text x={elbow.x + 22} y={elbow.y - 54}>q₂={q2}°</text>
        <text x={(origin.x + elbow.x) / 2 - 8} y={(origin.y + elbow.y) / 2 - 12}>l₁={l1.toFixed(2)} m</text>
        <text x={(elbow.x + tip.x) / 2 + 8} y={(elbow.y + tip.y) / 2 - 12}>l₂={l2.toFixed(2)} m</text>
        <text x={elbow.x - 28} y={elbow.y + 30}>E</text><text x={tip.x + 14} y={tip.y - 8}>P({values.x.toFixed(3)}, {values.y.toFixed(3)})</text>
        <text x={elbow.x + 45} y={elbow.y + 30}>{lt("绝对方向 q₁+q₂=")}{q1 + q2}°</text>
        <text className="kin-component-label" x={(origin.x + elbow.x) / 2} y={origin.y + 18}>l₁ cos q₁={values.x1.toFixed(3)}</text>
        <text className="kin-component-label" x={elbow.x + (tip.x - elbow.x) / 2} y={elbow.y + 18}>l₂ cos(q₁+q₂)={values.x2.toFixed(3)}</text>
        <text className="kin-component-label" x={origin.x + 8} y={(origin.y + elbow.y) / 2}>l₁ sin q₁={values.y1.toFixed(3)}</text>
        <text className="kin-component-label" x={elbow.x + 8} y={(elbow.y + tip.y) / 2}>l₂ sin(q₁+q₂)={values.y2.toFixed(3)}</text>
      </svg>
    </div>
    <div className="kinematics-controls">
      <label><span>q₁</span><input aria-label={lt("第一关节角 q1")} type="range" min="-80" max="140" value={q1} onChange={(event) => setQ1(Number(event.target.value))}/><output>{q1}°</output></label>
      <label><span>q₂</span><input aria-label={lt("第二关节相对角 q2")} type="range" min="-150" max="150" value={q2} onChange={(event) => setQ2(Number(event.target.value))}/><output>{q2}°</output></label>
      <button onClick={reset}>{lt("恢复示例")}</button>
    </div>
    <div className="kinematics-readout" aria-live="polite">
      <div><span>{lt("水平分量")}</span><strong>x = {values.x1.toFixed(3)} + {values.x2.toFixed(3)} = {values.x.toFixed(3)} m</strong></div>
      <div><span>{lt("竖直分量")}</span><strong>y = {values.y1.toFixed(3)} + {values.y2.toFixed(3)} = {values.y.toFixed(3)} m</strong></div>
      <code>x=l₁cos(q₁)+l₂cos(q₁+q₂) · y=l₁sin(q₁)+l₂sin(q₁+q₂)</code>
    </div>
    <figcaption><strong>{lt("二维二连杆正运动学与分量投影")}</strong><span>{lt("角度滑块使用真实三角公式实时计算；示例恢复到 l₁=0.40 m、l₂=0.30 m、q₁=30°、q₂=60°。")}</span><em>{lt("教学示意 · 本地 React SVG；公式计算而非预制动画")}</em></figcaption>
  </figure>;
}

export function LearningEmbeddedVisual({ kind }: { kind: string }) {
  if (kind === "arm-anatomy") return <RobotArmAnatomyDiagram/>;
  if (kind === "planar-kinematics") return <PlanarKinematicsDiagram/>;
  return null;
}
