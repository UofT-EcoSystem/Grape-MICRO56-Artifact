import threading
from abc import abstractmethod, ABC

this_thread = threading.local()
this_thread.context_mgr = {}


class _SingularContext(ABC):
    @abstractmethod
    def enter(self):
        pass

    def __enter__(self):
        assert (
            this_thread.context_mgr.get(type(self), None) is None
        ), "Nested context is not supported"
        this_thread.context_mgr[type(self)] = self
        self.enter()

    @abstractmethod
    def exit(self, *exception_args):
        pass

    def __exit__(self, *exception_args):
        self.exit(*exception_args)
        this_thread.context_mgr[type(self)] = None


class _NestableContext(ABC):
    __slots__ = ("_outer_scope_ctx",)

    def __init__(self):
        self._outer_scope_ctx = None

    @abstractmethod
    def enter(self):
        pass

    def __enter__(self):
        self._outer_scope_ctx = this_thread.context_mgr.get(type(self), None)
        this_thread.context_mgr[type(self)] = self
        self.enter()

    @abstractmethod
    def exit(self, *exception_args):
        pass

    def __exit__(self, *exception_args):
        self.exit(*exception_args)
        this_thread.context_mgr[type(self)] = self._outer_scope_ctx


class _RecoverableContext(_NestableContext, ABC):
    def __exit__(self, *exception_args):
        super().__exit__(*exception_args)
        current_ctx = this_thread.context_mgr.get(type(self), None)
        if current_ctx is not None:
            current_ctx.enter()
