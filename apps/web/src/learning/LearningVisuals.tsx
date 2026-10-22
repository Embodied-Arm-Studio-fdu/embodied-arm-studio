import { useLearningText } from "./learningTranslations";
import { useEffect, useMemo, useState } from "react";
import { Pause, Play, RotateCcw, StepForward } from "lucide-react";

const TITLES = [
  "环境状态 → 观测 → 决策 → 动作 → 环境变化 → 新观测", "S1机械臂关节与末端运动", "reset / step与开闭环控制", "小型神经网络前向传播",
  "MDP轨迹与折扣回报", "PPO裁剪目标函数", "Observation、Action、Reward联动", "训练曲线与正式评估", "泛化、迁移与领域偏移",
];

function DemoControls({ running, setRunning, reset, step, speed, setSpeed }: { running: boolean; setRunning: (value: boolean) => void; reset: () => void; step: () => void; speed: number; setSpeed: (value: number) => void }) {
  const lt = useLearningText();
  return <div className="learning-demo-controls">
    <button onClick={() => setRunning(true)} disabled={running}><Play size={14} />{lt("开始")}</button>
    <button onClick={() => setRunning(false)} disabled={!running}><Pause size={14} />{lt("暂停")}</button>
    <button onClick={reset}><RotateCcw size={14} />{lt("重置")}</button>
    <button onClick={step}><StepForward size={14} />{lt("单步")}</button>
    <label>{lt("速度")}<select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value={1400}>{lt("慢")}</option><option value={800}>{lt("标准")}</option><option value={400}>{lt("快")}</option></select></label>
  </div>;
}

function ArmDemo({ phase, value, setValue }: { phase: number; value: number; setValue: (value: number) => void }) {
  const lt = useLearningText();
  const q1 = value * Math.PI / 180;
  const q2 = ((phase * 12) - 45) * Math.PI / 180;
  const elbow = { x: 120 + 72 * Math.cos(q1), y: 112 - 72 * Math.sin(q1) };
  const tip = { x: elbow.x + 58 * Math.cos(q1 + q2), y: elbow.y - 58 * Math.sin(q1 + q2) };
  return <><svg viewBox="0 0 300 180" role="img" aria-label={lt("教学用二维机械臂关节和工作空间示意")}><path d="M50 112 A70 70 0 1 1 190 112" className="demo-workspace"/><line x1="120" y1="112" x2={elbow.x} y2={elbow.y}/><line x1={elbow.x} y1={elbow.y} x2={tip.x} y2={tip.y}/><circle cx="120" cy="112" r="8"/><circle cx={elbow.x} cy={elbow.y} r="7"/><circle className="demo-accent" cx={tip.x} cy={tip.y} r="6"/><text x="12" y="24">q₁={value}° · q₂={Math.round(q2 * 180 / Math.PI)}°</text><text x="12" y="43">{lt("末端 (")}{((tip.x - 120) / 100).toFixed(2)}, {((112 - tip.y) / 100).toFixed(2)}) m</text></svg><label className="demo-slider">{lt("关节角 q₁")}<input type="range" min="-70" max="110" value={value} onChange={(event) => setValue(Number(event.target.value))}/></label></>;
}

function PpoDemo({ value, setValue, epsilon }: { value: number; setValue: (value: number) => void; epsilon: number }) {
  const lt = useLearningText();
  const clipped = Math.max(1 - epsilon, Math.min(1 + epsilon, value));
  return <><svg viewBox="0 0 360 170" role="img" aria-label={lt("PPO概率比裁剪目标教学曲线")}><line x1="35" y1="135" x2="340" y2="135"/><line x1="35" y1="20" x2="35" y2="135"/><path d="M35 125 L340 30"/><path className="demo-accent-line" d={`M35 125 L${35 + (1 - epsilon) * 152} ${125 - (1 - epsilon) * 47} L${35 + (1 + epsilon) * 152} ${125 - (1 + epsilon) * 47} L340 ${125 - (1 + epsilon) * 47}`}/><circle className="demo-accent" cx={35 + value * 152} cy={125 - clipped * 47} r="6"/><text x="45" y="24">r={value.toFixed(2)} · ε={epsilon.toFixed(2)}</text><text x="45" y="43">{lt("未裁剪=")}{value.toFixed(2)}{lt("A · 裁剪=")}{clipped.toFixed(2)}{lt("A（A>0示意）")}</text></svg><label className="demo-slider">{lt("概率比 r")}<input type="range" min="0" max="200" value={value * 100} onChange={(event) => setValue(Number(event.target.value) / 100)}/></label></>;
}

function ControlLoopDemo({ phase }: { phase: number }) {
  const lt = useLearningText();
  const [closedLoop, setClosedLoop] = useState(true);
  const [noise, setNoise] = useState(12);
  const [delay, setDelay] = useState(1);
  const error = Math.max(4, 72 - phase * (closedLoop ? 9 : 3) + noise * .25 + delay * 3);
  return <><svg viewBox="0 0 520 170" role="img" aria-label={lt("reset step开环与闭环误差示意")}><path d="M30 30 V140 H500"/><path className="demo-target-line" d="M30 55 H500"/><path className="demo-accent-line" d={`M30 132 Q150 ${105 + delay * 5} 250 ${70 + error / 3} T500 ${55 + error / 4}`}/><text x="42" y="48">{lt("目标")}</text><text x="360" y="145">step {phase}</text><text x="42" y="126">{lt("当前误差")} {error.toFixed(1)}</text></svg><div className="demo-options"><label>{lt("控制方式")}<select value={closedLoop ? "closed" : "open"} onChange={(event) => setClosedLoop(event.target.value === "closed")}><option value="closed">{lt("闭环：按误差修正")}</option><option value="open">{lt("开环：不读取反馈")}</option></select></label><label>{lt("观测噪声")} {noise}%<input type="range" min="0" max="40" value={noise} onChange={(event) => setNoise(Number(event.target.value))}/></label><label>{lt("动作延迟")} {delay} {lt("步")}<input type="range" min="0" max="4" value={delay} onChange={(event) => setDelay(Number(event.target.value))}/></label></div></>;
}

function NetworkDemo({ value, setValue, phase }: { value: number; setValue: (value: number) => void; phase: number }) {
  const lt = useLearningText();
  const [weight, setWeight] = useState(0.8);
  const [activation, setActivation] = useState("relu");
  const x = value / 25 - 2;
  const z = x * weight + .2;
  const output = activation === "tanh" ? Math.tanh(z) : activation === "linear" ? z : Math.max(0, z);
  return <><svg viewBox="0 0 520 170" role="img" aria-label={lt("小型神经网络逐层前向传播示意")}><g className={phase % 4 === 0 ? "active" : ""}><circle cx="70" cy="82" r="25"/><text x="70" y="86" textAnchor="middle">x={x.toFixed(2)}</text></g><path d="M96 82 H185"/><text x="132" y="70">× w={weight.toFixed(1)}</text><g className={phase % 4 === 1 ? "active" : ""}><circle cx="220" cy="82" r="31"/><text x="220" y="86" textAnchor="middle">z={z.toFixed(2)}</text></g><path d="M252 82 H335"/><text x="275" y="70">{activation}</text><g className={phase % 4 >= 2 ? "active" : ""}><circle cx="382" cy="82" r="32"/><text x="382" y="86" textAnchor="middle">y={output.toFixed(2)}</text></g></svg><div className="demo-options"><label>{lt("输入 x")}<input type="range" min="0" max="100" value={value} onChange={(event) => setValue(Number(event.target.value))}/></label><label>{lt("权重 w")} {weight.toFixed(1)}<input type="range" min="-20" max="20" value={weight * 10} onChange={(event) => setWeight(Number(event.target.value) / 10)}/></label><label>{lt("激活函数")}<select value={activation} onChange={(event) => setActivation(event.target.value)}><option value="relu">ReLU</option><option value="tanh">tanh</option><option value="linear">{lt("线性")}</option></select></label></div></>;
}

function MdpDemo({ value, setValue, phase }: { value: number; setValue: (value: number) => void; phase: number }) {
  const lt = useLearningText();
  const [action, setAction] = useState("safe");
  const gamma = value / 100;
  const rewards = action === "safe" ? [1, 1, 2, 3] : [-1, 0, 2, 8];
  const discounted = rewards.reduce((total, reward, index) => total + reward * gamma ** index, 0);
  return <><svg viewBox="0 0 520 170" role="img" aria-label={lt("MDP状态动作奖励轨迹和折扣权重")}><line x1="60" y1="72" x2="455" y2="72"/>{rewards.map((reward, index) => <g key={index} className={index === phase % rewards.length ? "active" : ""}><circle cx={65 + index * 125} cy="72" r="23"/><text x={65 + index * 125} y="76" textAnchor="middle">s{index}</text><text x={105 + index * 125} y="53">r={reward}</text><rect x={48 + index * 125} y={125 - Math.abs(reward) * 7} width="34" height={Math.abs(reward) * 7}/></g>)}<text x="300" y="155">{lt("折扣回报 G=")}{discounted.toFixed(2)}</text></svg><div className="demo-options"><label>{lt("动作策略")}<select value={action} onChange={(event) => setAction(event.target.value)}><option value="safe">{lt("稳健路径")}</option><option value="risky">{lt("高风险远期奖励")}</option></select></label><label>{lt("折扣因子 γ=")}{gamma.toFixed(2)}<input type="range" min="0" max="100" value={value} onChange={(event) => setValue(Number(event.target.value))}/></label></div></>;
}

function PpoInteractive({ value, setValue }: { value: number; setValue: (value: number) => void }) {
  const lt = useLearningText();
  const [advantage, setAdvantage] = useState(1);
  const [epsilon, setEpsilon] = useState(.2);
  const ratio = value / 50;
  const unclipped = ratio * advantage;
  const clippedRatio = Math.max(1 - epsilon, Math.min(1 + epsilon, ratio));
  const objective = Math.min(unclipped, clippedRatio * advantage);
  return <><PpoDemo value={ratio} setValue={(next) => setValue(next * 50)} epsilon={epsilon}/><div className="demo-readout">{lt("未裁剪")} {unclipped.toFixed(2)} {lt("· 裁剪后目标")} {objective.toFixed(2)} {lt("· 裁剪范围 [")}{(1 - epsilon).toFixed(2)}, {(1 + epsilon).toFixed(2)}]</div><div className="demo-options"><label>{lt("优势 A=")}{advantage.toFixed(1)}<input type="range" min="-20" max="20" value={advantage * 10} onChange={(event) => setAdvantage(Number(event.target.value) / 10)}/></label><label>ε={epsilon.toFixed(2)}<input type="range" min="5" max="40" value={epsilon * 100} onChange={(event) => setEpsilon(Number(event.target.value) / 100)}/></label></div></>;
}

const TASK_CONTRACTS = {
  Reach: { observation: "18D · 关节/末端/目标", action: "二维归一化末端 XY", terms: ["距离", "进展", "动作代价"] },
  Push: { observation: "24D · 末端/物体/目标", action: "二维平面位移", terms: ["目标距离", "方向进展", "接触"] },
  Grasp: { observation: "40D · 位姿/夹爪/接触", action: "XYZ + 夹爪", terms: ["接近", "闭合", "抬升验证"] },
};

function TaskContractDemo({ value, setValue }: { value: number; setValue: (value: number) => void }) {
  const lt = useLearningText();
  const [task, setTask] = useState<keyof typeof TASK_CONTRACTS>("Reach");
  const contract = TASK_CONTRACTS[task];
  const weights = [value / 100, .65, .35];
  return <><div className="demo-contract"><div><span>Observation</span><strong>{lt(contract.observation)}</strong></div><div><span>Action</span><strong>{lt(contract.action)}</strong></div><div><span>Reward</span>{contract.terms.map((term, index) => <p key={term}><i style={{ width: `${weights[index] * 100}%` }}/><b>{lt(term)} · {weights[index].toFixed(2)}</b></p>)}</div></div><div className="demo-options"><label>{lt("任务")}<select value={task} onChange={(event) => setTask(event.target.value as keyof typeof TASK_CONTRACTS)}>{Object.keys(TASK_CONTRACTS).map((name) => <option key={name}>{name}</option>)}</select></label><label>{lt("主奖励权重")} {weights[0].toFixed(2)}<input type="range" min="0" max="100" value={value} onChange={(event) => setValue(Number(event.target.value))}/></label></div><p className="demo-readout">{lt("行为倾向说明：提高主奖励权重会改变优化信号，但不等同于正式成功率提高。")}</p></>;
}

function EvaluationDemo({ phase }: { phase: number }) {
  const lt = useLearningText();
  const seeds = [{ p: 88, lo: 78, hi: 94 }, { p: 72, lo: 60, hi: 82 }, { p: 91, lo: 82, hi: 96 }];
  return <><svg viewBox="0 0 520 210" role="img" aria-label={lt("奖励曲线成功率与多seed Wilson区间教学数据")}><polyline className="demo-accent-line" points="35,120 95,108 155,92 215,79 275,62"/><polyline points="35,130 95,130 155,115 215,115 275,90"/><text x="35" y="32">{lt("教学数据：奖励持续上升，成功率分段变化")}</text><text x="285" y="65">reward</text><text x="285" y="94">success rate</text>{seeds.map((seed, index) => <g key={index} className={index === phase % seeds.length ? "active" : ""}><line x1={350 + index * 55} x2={350 + index * 55} y1={180 - seed.lo} y2={180 - seed.hi}/><circle cx={350 + index * 55} cy={180 - seed.p} r="6"/><text x={350 + index * 55} y="194" textAnchor="middle">seed {index + 1}</text></g>)}</svg><div className="demo-readout">{lt("点为各训练 seed 的独立评估成功率，竖线为 Wilson 95% 区间；还需结合 timeout、明确 failure 与代表性失败案例。")}</div></>;
}

function TransferDemo() {
  const lt = useLearningText();
  const [position, setPosition] = useState("seen");
  const [mass, setMass] = useState("1.0");
  const [friction, setFriction] = useState("0.5");
  const [seed, setSeed] = useState("train");
  const [mode, setMode] = useState("resume");
  const shifted = position !== "seen" || mass !== "1.0" || friction !== "0.5" || seed !== "train";
  const descriptions: Record<string, string> = { resume: lt("checkpoint恢复：继续同一任务与兼容配置"), transfer: lt("迁移初始化：复用参数作为新任务起点"), finetune: lt("微调：在新数据或新条件下继续更新参数") };
  return <><div className={`demo-domain ${shifted ? "shifted" : ""}`}><div><span>{lt("训练域")}</span><strong>{lt("位置 A · 质量 1.0 · 摩擦 0.5 · 训练 seed")}</strong></div><div><span>{lt("测试域")}</span><strong>{position} {lt("· 质量")} {mass} {lt("· 摩擦")} {friction} · {seed}</strong></div><p>{shifted ? lt("检测到领域偏移：必须重新评估，不能沿用原成功结论。") : lt("条件与训练域一致；仍需独立评估。")}</p></div><div className="demo-options"><label>{lt("物体位置")}<select value={position} onChange={(event) => setPosition(event.target.value)}><option value="seen">{lt("已见位置 A")}</option><option value="unseen">{lt("未见位置 B")}</option></select></label><label>{lt("质量")}<select value={mass} onChange={(event) => setMass(event.target.value)}><option>1.0</option><option>1.5</option></select></label><label>{lt("摩擦")}<select value={friction} onChange={(event) => setFriction(event.target.value)}><option>0.5</option><option>0.2</option><option>0.8</option></select></label><label>seed<select value={seed} onChange={(event) => setSeed(event.target.value)}><option value="train">{lt("训练 seed")}</option><option value="unseen">{lt("未见 seed")}</option></select></label><label>{lt("参数使用方式")}<select value={mode} onChange={(event) => setMode(event.target.value)}><option value="resume">{lt("checkpoint恢复")}</option><option value="transfer">{lt("迁移初始化")}</option><option value="finetune">{lt("微调")}</option></select></label></div><p className="demo-readout">{descriptions[mode]}</p></>;
}

const LOOP_PHASES = ["环境状态", "观测", "决策", "动作", "环境变化", "新观测"];

function EmbodiedLoopDemo({ phase }: { phase: number }) {
  const lt = useLearningText();
  const active = phase % LOOP_PHASES.length;
  return <><div className="embodied-loop-stage" aria-live="polite"><span>{lt("当前阶段")}</span><strong>{String(active + 1).padStart(2, "0")} · {lt(LOOP_PHASES[active])}</strong><p>{[
    lt("环境包含机器人、物体和任务条件。"),
    lt("传感器把环境的一部分转换为策略可用的观测。"),
    lt("策略或规划器依据当前观测选择下一步。"),
    lt("执行器把决策转换为机械臂动作。"),
    lt("动作通过物理交互改变机器人与环境。"),
    lt("系统读取变化后的新观测，闭环进入下一轮。"),
  ][active]}</p></div><svg viewBox="0 0 720 240" role="img" aria-label={lt("环境状态到新观测的具身交互闭环教学示意")}><defs><marker id="loop-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z"/></marker></defs>{LOOP_PHASES.map((label, index) => { const x = 22 + index * 115; const y = index % 2 === 0 ? 52 : 112; return <g key={label} className={index === active ? "active" : ""}><rect x={x} y={y} width="98" height="56" rx="8"/><text x={x + 49} y={y + 34} textAnchor="middle">{lt(label)}</text>{index < LOOP_PHASES.length - 1 && <path markerEnd="url(#loop-arrow)" d={`M${x + 98} ${y + 28} C${x + 108} ${y + 28}, ${x + 108} ${(index + 1) % 2 === 0 ? 80 : 140}, ${x + 114} ${(index + 1) % 2 === 0 ? 80 : 140}`}/>}</g>; })}<path className="demo-feedback-loop" markerEnd="url(#loop-arrow)" d="M695 140 C704 210 74 225 40 112"/><text x="326" y="221">{lt("反馈使下一次决策使用更新后的观测")}</text></svg></>;
}

function GenericDemo({ chapter, phase, value, setValue }: { chapter: number; phase: number; value: number; setValue: (value: number) => void }) {
  if (chapter === 1) return <EmbodiedLoopDemo phase={phase}/>;
  if (chapter === 2) return <ArmDemo phase={phase} value={value} setValue={setValue}/>;
  if (chapter === 3) return <ControlLoopDemo phase={phase}/>;
  if (chapter === 4) return <NetworkDemo value={value} setValue={setValue} phase={phase}/>;
  if (chapter === 5) return <MdpDemo value={value} setValue={setValue} phase={phase}/>;
  if (chapter === 6) return <PpoInteractive value={value} setValue={setValue}/>;
  if (chapter === 7) return <TaskContractDemo value={value} setValue={setValue}/>;
  if (chapter === 8) return <EvaluationDemo phase={phase}/>;
  if (chapter === 9) return <TransferDemo/>;
  return null;
}

export function LearningVisual({ chapter }: { chapter: number }) {
  const lt = useLearningText();
  const reducedMotion = useMemo(() => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches, []);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState(0);
  const [speed, setSpeed] = useState(800);
  const [value, setValue] = useState(chapter === 2 ? 25 : 60);
  const phaseLimit = chapter === 1 ? LOOP_PHASES.length - 1 : 8;
  useEffect(() => {
    if (!running || reducedMotion) return;
    const timer = window.setInterval(() => setPhase((current) => Math.min(current + 1, phaseLimit)), speed);
    return () => window.clearInterval(timer);
  }, [running, speed, reducedMotion, phaseLimit]);
  useEffect(() => { if (phase >= phaseLimit) setRunning(false); }, [phase, phaseLimit]);
  const reset = () => { setRunning(false); setPhase(0); setValue(chapter === 2 ? 25 : 60); };
  if (chapter === 2 || chapter >= 10) return null;
  return <section className="learning-demo" aria-labelledby={`demo-title-${chapter}`}>
    <header><div><span>{lt("核心演示 · 教学示意")}</span><h3 id={`demo-title-${chapter}`}>{lt(TITLES[chapter - 1])}</h3></div><p>{lt("用于解释概念关系，不代表真实训练、仿真结果或实物控制。")}</p></header>
    <GenericDemo chapter={chapter} phase={phase} value={value} setValue={setValue}/>
    <DemoControls running={running} setRunning={setRunning} reset={reset} step={() => { setRunning(false); setPhase((current) => Math.min(current + 1, phaseLimit)); }} speed={speed} setSpeed={setSpeed}/>
    {reducedMotion && <small>{lt("已遵循系统“减少动态效果”设置；可使用“单步”查看过程。")}</small>}
  </section>;
}

