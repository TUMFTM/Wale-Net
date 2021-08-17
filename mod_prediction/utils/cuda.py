def cudanize(hist, nbrs, fut, sc_img):
    """This script takes the network input tensors and creates the tensors on cuda."""
    if hist is not None:
        hist = hist.cuda()
    if nbrs is not None:
        nbrs = nbrs.cuda()
    if fut is not None:
        fut = fut.cuda()
    if sc_img is not None:
        sc_img = sc_img.cuda()

    return hist, nbrs, fut, sc_img
