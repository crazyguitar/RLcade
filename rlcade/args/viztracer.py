"""VizTracer profiling argument group."""

# fmt: off
def add_viztracer_args(parser):
    group = parser.add_argument_group("viztracer", "VizTracer profiling options")
    group.add_argument("--viztracer", type=str, default=None, help="VizTracer output file (enables profiling)")
    group.add_argument("--viztracer-start", type=int, default=1, help="Start profiling at this step")
    group.add_argument("--viztracer-end", type=int, default=10, help="Stop profiling at this step")
    group.add_argument("--viztracer-tracer-entries", type=int, default=1000000, help="Max tracer entries")
    group.add_argument("--viztracer-max-stack-depth", type=int, default=-1, help="Max stack depth (-1 for unlimited)")
    group.add_argument("--viztracer-include-files", nargs="*", default=None, help="Only trace these files")
    group.add_argument("--viztracer-exclude-files", nargs="*", default=None, help="Exclude these files from tracing")
    group.add_argument("--viztracer-ignore-c-function", action="store_true", help="Ignore C function calls")
    group.add_argument("--viztracer-ignore-frozen", action="store_true", help="Ignore frozen modules")
    group.add_argument("--viztracer-log-func-retval", action="store_true", help="Log function return values")
    group.add_argument("--viztracer-log-func-args", action="store_true", help="Log function arguments")
    group.add_argument("--viztracer-log-print", action="store_true", help="Log print() calls")
    group.add_argument("--viztracer-log-gc", action="store_true", help="Log garbage collection")
    group.add_argument("--viztracer-log-sparse", action="store_true", help="Enable sparse logging mode")
    group.add_argument("--viztracer-log-async", action="store_true", help="Log async activities")
    group.add_argument("--viztracer-log-torch", action="store_true", help="Log PyTorch operator calls")
    group.add_argument("--viztracer-min-duration", type=float, default=0, help="Min duration (us) to record")
    group.add_argument("--viztracer-minimize-memory", action="store_true", help="Minimize memory usage")
    group.add_argument("--viztracer-process-name", type=str, default=None, help="Set process name in trace")
# fmt: on
