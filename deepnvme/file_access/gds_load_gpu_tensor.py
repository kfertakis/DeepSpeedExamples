import torch
import os, timeit, functools
from utils import parse_read_arguments
from deepspeed.ops.op_builder import GDSBuilder

def file_read(inp_f, h, gpu_buffer):
    h.sync_pread(gpu_buffer, inp_f)
    return gpu_buffer.cuda()

def main():
    args = parse_read_arguments()
    input_file = args.input_file
    file_sz = os.path.getsize(input_file)
    cnt = args.loop

    gds_handle = GDSBuilder().load().gds_handle(1024**2, 128, True, True, 1)
    gds_buffer = torch.empty(file_sz, dtype=torch.uint8, device='cuda', requires_grad=False)

    t = timeit.Timer(functools.partial(file_read, input_file, gds_handle, gds_buffer))
    gds_t = t.timeit(cnt)
    gds_gbs = (cnt*os.path.getsize(input_file))/gds_t/1e9
    print(f'gds load_gpu: {file_sz/(1024**3)}GB, {gds_gbs:5.2f} GB/sec, {gds_t:5.2f} secs')

    if args.validate: 
        from py_load_cpu_tensor import file_read as py_file_read 
        aio_tensor = file_read(input_file, gds_handle, gds_buffer).cpu()
        py_tensor = py_file_read(input_file)
        print(f'Validation success = {aio_tensor.equal(py_tensor)}')

if __name__ == "__main__":
    main()
