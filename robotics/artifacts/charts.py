import io
import math
from threading import Lock


_RENDER_LOCK = Lock()


def metric_chart_png(events, metric):
    points = {}
    for event in events:
        if event.type.value != "metric":
            continue
        metrics = event.payload.get("metrics", {})
        update = metrics.get("update_index")
        value = metrics.get(metric)
        if type(update) is int and update > 0 and type(value) in (float, int) and math.isfinite(value):
            points[update] = value
    if not points:
        raise ValueError("No recorded numeric data for this metric")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with _RENDER_LOCK:
        figure = Figure(figsize=(10, 4), dpi=160, facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        updates = sorted(points)
        axis.plot(updates, [points[update] for update in updates], color="#287e6f", marker="o", markersize=3)
        axis.set_xlabel("Update")
        axis.set_ylabel(metric)
        axis.set_title(metric)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png")
        figure.clear()
        return output.getvalue()