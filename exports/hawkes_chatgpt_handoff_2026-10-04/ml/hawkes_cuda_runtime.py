"""Compile CUDA device code with the NVRTC runtime bundled with PyTorch."""
from __future__ import annotations
import ctypes as ct
import os
from pathlib import Path
import torch


class CudaKernel:
    def __init__(self, source: str, name: str):
        torch.cuda.init()
        self.context_anchor = torch.empty(1, device='cuda')
        lib_directory = Path(torch.__file__).parent / 'lib'
        self.dll_directory = os.add_dll_directory(str(lib_directory)) if os.name == 'nt' else None
        candidates = list(lib_directory.glob('nvrtc64_*.dll')) if os.name == 'nt' else list(lib_directory.glob('libnvrtc.so*'))
        candidates = [p for p in candidates if '.alt.' not in p.name]
        if not candidates:
            raise RuntimeError('Bundled NVRTC compiler unavailable')
        self.nvrtc = ct.WinDLL(str(candidates[0])) if os.name == 'nt' else ct.CDLL(str(candidates[0]))
        self.driver = ct.WinDLL('nvcuda.dll') if os.name == 'nt' else ct.CDLL('libcuda.so.1')
        def declare(library, symbol, args):
            function = getattr(library, symbol)
            function.argtypes, function.restype = args, ct.c_int
            return function
        pointer = ct.c_void_p
        self.create = declare(self.nvrtc, 'nvrtcCreateProgram', [ct.POINTER(pointer), ct.c_char_p, ct.c_char_p, ct.c_int, pointer, pointer])
        self.compile = declare(self.nvrtc, 'nvrtcCompileProgram', [pointer, ct.c_int, ct.POINTER(ct.c_char_p)])
        self.log_size = declare(self.nvrtc, 'nvrtcGetProgramLogSize', [pointer, ct.POINTER(ct.c_size_t)])
        self.log = declare(self.nvrtc, 'nvrtcGetProgramLog', [pointer, pointer])
        self.ptx_size = declare(self.nvrtc, 'nvrtcGetPTXSize', [pointer, ct.POINTER(ct.c_size_t)])
        self.ptx = declare(self.nvrtc, 'nvrtcGetPTX', [pointer, pointer])
        self.destroy = declare(self.nvrtc, 'nvrtcDestroyProgram', [ct.POINTER(pointer)])
        self.module_load = declare(self.driver, 'cuModuleLoadData', [ct.POINTER(pointer), pointer])
        self.module_function = declare(self.driver, 'cuModuleGetFunction', [ct.POINTER(pointer), pointer, ct.c_char_p])
        self.launch = declare(self.driver, 'cuLaunchKernel', [pointer, *([ct.c_uint] * 7), pointer, pointer, pointer])
        program = pointer()
        self.check(self.create(ct.byref(program), source.encode(), b'hawkes.cu', 0, None, None), 'create NVRTC program')
        major, minor = torch.cuda.get_device_capability()
        options = [f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--fmad=false', b'--std=c++17']
        try:
            result = self.compile(program, len(options), (ct.c_char_p * len(options))(*options))
            size = ct.c_size_t()
            self.check(self.log_size(program, ct.byref(size)), 'read compile log size')
            log = ct.create_string_buffer(size.value)
            self.check(self.log(program, log), 'read compile log')
            if result:
                raise RuntimeError(f'NVRTC compilation failed ({result}):\n{log.value.decode()}')
            self.check(self.ptx_size(program, ct.byref(size)), 'read PTX size')
            code = ct.create_string_buffer(size.value)
            self.check(self.ptx(program, code), 'read PTX')
            self.module, self.function = pointer(), pointer()
            self.check(self.module_load(ct.byref(self.module), code), 'load CUDA module')
            self.check(self.module_function(ct.byref(self.function), self.module, name.encode()), 'find CUDA kernel')
        finally:
            self.destroy(ct.byref(program))

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f'{operation}: CUDA/NVRTC error {code}')

    def __call__(self, tensors, integers, blocks, threads):
        values = [ct.c_void_p(t.data_ptr()) for t in tensors] + [ct.c_int(i) for i in integers]
        arguments = (ct.c_void_p * len(values))(*(ct.addressof(value) for value in values))
        self.check(self.launch(self.function, blocks, 1, 1, threads, 1, 1, 0,
                               ct.c_void_p(torch.cuda.current_stream().cuda_stream), arguments, None), 'launch CUDA kernel')
