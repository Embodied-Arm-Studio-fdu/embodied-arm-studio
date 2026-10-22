import { useEffect, useRef, useState, type PointerEvent } from "react";

const STORAGE_KEY = "embodied-arm-agent-position";
export type AgentPosition = { x: number; y: number };
export function clampAgentPosition(point: AgentPosition, width: number, height: number, bottomInset = 0): AgentPosition {
  return { x: Math.max(12, Math.min(point.x, width - 64)), y: Math.max(12, Math.min(point.y, height - 64 - bottomInset)) };
}

export function useFloatingAgentPosition(bottomInset = 0) {
  const [position, setPosition] = useState<AgentPosition>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
      if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) return clampAgentPosition(saved, window.innerWidth, window.innerHeight, bottomInset);
    } catch { /* Use the default when saved coordinates are invalid. */ }
    return clampAgentPosition({ x: window.innerWidth - 76, y: window.innerHeight - 148 }, window.innerWidth, window.innerHeight, bottomInset);
  });
  const drag = useRef<{ pointerId: number; x: number; y: number; origin: AgentPosition; moved: boolean } | null>(null);
  const suppressClick = useRef(false);
  useEffect(() => {
    const resize = () => setPosition((current) => clampAgentPosition(current, window.innerWidth, window.innerHeight, bottomInset));
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [bottomInset]);
  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(position)); }
    catch { /* Dragging remains usable when browser storage is unavailable. */ }
  }, [position]);
  function onPointerDown(event: PointerEvent<HTMLButtonElement>) {
    if (event.button !== 0) return;
    suppressClick.current = false;
    drag.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, origin: position, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function onPointerMove(event: PointerEvent<HTMLButtonElement>) {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    const dx = event.clientX - current.x, dy = event.clientY - current.y;
    if (!current.moved && Math.hypot(dx, dy) < 5) return;
    current.moved = true;
    const next = clampAgentPosition({ x: current.origin.x + dx, y: current.origin.y + dy }, window.innerWidth, window.innerHeight, bottomInset);
    setPosition((previous) => previous.x === next.x && previous.y === next.y ? previous : next);
  }
  function onPointerUp(event: PointerEvent<HTMLButtonElement>) {
    if (!drag.current || drag.current.pointerId !== event.pointerId) return;
    suppressClick.current = drag.current.moved;
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }
  const panelWidth = Math.min(380, window.innerWidth - 24);
  const panelHeight = Math.min(560, window.innerHeight - 92);
  const panelLeft = Math.max(12, Math.min(position.x + 48 - panelWidth, window.innerWidth - panelWidth - 12));
  const panelTop = Math.max(12, Math.min(position.y - panelHeight - 10, window.innerHeight - panelHeight - 12));
  return { position, panelStyle: { left: panelLeft, top: panelTop, width: panelWidth, height: panelHeight },
    handlers: { onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp },
    consumeDragClick: () => { const moved = suppressClick.current; suppressClick.current = false; return moved; } };
}
