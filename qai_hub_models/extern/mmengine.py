from contextlib import contextmanager

from mmengine.runner import checkpoint


@contextmanager
def patch_mmengine_torch_load_no_weights_only():
    """
    Within this context manager, weights_only=False is always passed to torch.load()
    for compatibility with torch 2.6 and newer.
    """
    load_url = checkpoint.load_url

    def load_url_no_weights_only(*args, **kwargs):
        return load_url(*args, **kwargs, weights_only=False)

    checkpoint.load_url = load_url_no_weights_only

    torch_load = checkpoint.torch.load

    def torch_load_no_weights_only(*args, **kwargs):
        return torch_load(*args, **kwargs, weights_only=False)

    checkpoint.torch.load = torch_load_no_weights_only

    try:
        yield
    finally:
        checkpoint.load_url = load_url
        checkpoint.torch.load = torch_load
