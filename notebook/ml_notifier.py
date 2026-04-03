"""
ml_notifier.py — Notificaciones push para experimentos de ML via ntfy.sh

Setup:
    1. Instala la app "ntfy" en tu celular (Play Store / F-Droid / App Store)
    2. Abre la app y suscríbete al topic que elijas (ej: "aaron-ml-lab-2026")
    3. Copia este archivo al directorio de tu proyecto
    4. En tu notebook:

        from ml_notifier import MLNotifier
        notifier = MLNotifier(topic="aaron-ml-lab-2026")

        # Como decorador
        @notifier.watch
        def train():
            ...
            return {"accuracy": 0.95, "f1": 0.91}

        # Como context manager
        with notifier.track("Experimento K-Means"):
            model.fit(X)

        # Llamada directa
        notifier.send("Mensaje libre")
"""

import time
import traceback
import functools
from datetime import timedelta

try:
    import requests
except ImportError:
    raise ImportError("Instala requests: pip install requests")


class MLNotifier:
    """Notificador push para experimentos de ML usando ntfy.sh"""

    NTFY_URL = "https://ntfy.sh"

    def __init__(self, topic: str, priority: str = "high", tags_ok: str = "white_check_mark", tags_fail: str = "x"):
        """
        Args:
            topic:     Nombre único de tu canal ntfy (ej: 'aaron-ml-lab-2026').
                       Usa algo difícil de adivinar para evitar spam.
            priority:  Prioridad por defecto ('min', 'low', 'default', 'high', 'urgent').
            tags_ok:   Emoji tag para éxito (sintaxis ntfy, sin dos puntos).
            tags_fail: Emoji tag para error.
        """
        self.topic = topic
        self.priority = priority
        self.tags_ok = tags_ok
        self.tags_fail = tags_fail

    # ------------------------------------------------------------------ #
    #  Core: enviar notificación                                          #
    # ------------------------------------------------------------------ #
    def send(self, message: str, title: str = "ML Pipeline", priority: str | None = None, tags: str | None = None):
        """Envía una notificación push.

        Args:
            message:  Cuerpo del mensaje.
            title:    Título de la notificación.
            priority: Sobreescribe la prioridad por defecto.
            tags:     Emoji tags de ntfy (ej: 'rocket,white_check_mark').
        """
        headers = {
            "Title": title,
            "Priority": priority or self.priority,
        }
        if tags:
            headers["Tags"] = tags

        try:
            resp = requests.post(
                f"{self.NTFY_URL}/{self.topic}",
                data=message.encode("utf-8"),
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            # No interrumpir el flujo del notebook por un fallo de notificación
            print(f"[MLNotifier] Error al enviar notificación: {e}")

    # ------------------------------------------------------------------ #
    #  Helpers para formatear resultados                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Convierte segundos a formato legible."""
        td = timedelta(seconds=int(seconds))
        parts = []
        hours, remainder = divmod(td.seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if td.days:
            parts.append(f"{td.days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)

    @staticmethod
    def _format_metrics(result) -> str:
        """Intenta extraer métricas del resultado de una función."""
        if result is None:
            return ""
        if isinstance(result, dict):
            lines = [f"  {k}: {v}" for k, v in result.items()]
            return "\n" + "\n".join(lines)
        return f"\n  Resultado: {result}"

    # ------------------------------------------------------------------ #
    #  Context manager: with notifier.track("nombre")                    #
    # ------------------------------------------------------------------ #
    class _Tracker:
        """Context manager interno."""

        def __init__(self, notifier: "MLNotifier", name: str):
            self.notifier = notifier
            self.name = name
            self.start = None

        def __enter__(self):
            self.start = time.time()
            print(f"[MLNotifier] Rastreando: {self.name}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start
            tiempo = MLNotifier._format_time(elapsed)

            if exc_type is None:
                self.notifier.send(
                    message=f"{self.name}\nTiempo: {tiempo}",
                    title="Ejecución completada",
                    tags=self.notifier.tags_ok,
                )
            else:
                tb_short = "".join(traceback.format_exception_only(exc_type, exc_val)).strip()
                self.notifier.send(
                    message=f"{self.name}\nTiempo: {tiempo}\nError: {tb_short}",
                    title="Ejecución fallida",
                    tags=self.notifier.tags_fail,
                    priority="urgent",
                )
            # No suprimir la excepción
            return False

    def track(self, name: str = "Experimento"):
        """Context manager que notifica al terminar (éxito o error).

        Uso:
            with notifier.track("K-Means clustering"):
                model.fit(X)
        """
        return self._Tracker(self, name)

    # ------------------------------------------------------------------ #
    #  Decorador: @notifier.watch                                        #
    # ------------------------------------------------------------------ #
    def watch(self, func=None, *, name: str | None = None):
        """Decorador que notifica cuando una función termina.

        Si la función retorna un dict, sus keys se incluyen como métricas.

        Uso:
            @notifier.watch
            def train_model():
                ...
                return {"accuracy": 0.95}

            @notifier.watch(name="Entrenamiento GMM")
            def train_gmm():
                ...
        """

        def decorator(fn):
            label = name or fn.__name__

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = fn(*args, **kwargs)
                    elapsed = time.time() - start
                    tiempo = self._format_time(elapsed)
                    metrics = self._format_metrics(result)
                    self.send(
                        message=f"{label}\nTiempo: {tiempo}{metrics}",
                        title="Ejecución completada",
                        tags=self.tags_ok,
                    )
                    return result
                except Exception as e:
                    elapsed = time.time() - start
                    tiempo = self._format_time(elapsed)
                    tb_short = "".join(traceback.format_exception_only(type(e), e)).strip()
                    self.send(
                        message=f"{label}\nTiempo: {tiempo}\nError: {tb_short}",
                        title="Ejecución fallida",
                        tags=self.tags_fail,
                        priority="urgent",
                    )
                    raise  # Re-lanza la excepción original

            return wrapper

        # Soportar @notifier.watch y @notifier.watch(name="...")
        if func is not None:
            return decorator(func)
        return decorator


# ------------------------------------------------------------------ #
#  Ejemplo rápido (ejecutar directamente para probar)                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    topic = sys.argv[1] if len(sys.argv) > 1 else "test-ml-notifier"
    n = MLNotifier(topic=topic)
    n.send("Prueba de conexión desde ml_notifier.py", title="Test")
    print(f"Notificación enviada al topic: {topic}")
