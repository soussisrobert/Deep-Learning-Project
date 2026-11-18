"""Generate Wave2 empty-positive guard patch; never apply it to upstream."""
from difflib import unified_diff
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "pysot/datasets/anchortarget.py"


def corrected_source(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError("pinned target source changed")
        source = source.replace(old, new, 1)

    replace("        labelcls1=np.zeros((1,size,size))-1", '''        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer")
        if size != cfg.TRAIN.OUTPUTFEATURE_SIZE:
            raise ValueError("size must equal configured OUTPUTFEATURE_SIZE")
        if (isinstance(cfg.TRAIN.SEARCH_SIZE, bool) or not isinstance(cfg.TRAIN.SEARCH_SIZE, (int, np.integer))
                or not np.isfinite([cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.LABEL_RANGE]).all()
                or cfg.TRAIN.SEARCH_SIZE < 2 or cfg.TRAIN.LABEL_RANGE <= 0
                or cfg.TRAIN.LABEL_RANGE > np.sqrt(np.finfo(float).max)/(4*size)):
            raise ValueError("invalid search size or LABEL_RANGE")
        target_check = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            extent = target_check[2:] - target_check[:2]
        if (not np.isfinite(target_check).all() or not np.isfinite(extent).all() or np.any(extent <= 0)
                or np.any(np.abs(target_check) > np.finfo(np.float32).max/4)):
            raise ValueError("bbox must have finite positive extents representable by label encoding")
        if (target_check[2] <= 0 or target_check[3] <= 0 or
                target_check[0] >= cfg.TRAIN.SEARCH_SIZE or target_check[1] >= cfg.TRAIN.SEARCH_SIZE):
            raise ValueError("positive bbox has no overlap with the search crop")
        labelcls1=np.zeros((1,size,size))-1''')
    replace('''        weightxff[np.where(weightxff<((w//2+h//2)*range/1.5)**2)]=1-((se-se.min())/(se.max()-se.min()+1e-4))
\x20\x20\x20\x20\x20\x20\x20\x20
        weightxff[np.where(weightxff>((w//2+h//2)*range/1.5)**2)]=0''', '''        if se.size:
            weightxff[np.where(weightxff<((w//2+h//2)*range/1.5)**2)]=1-((se-se.min())/(se.max()-se.min()+1e-4))
            weightxff[np.where(weightxff>((w//2+h//2)*range/1.5)**2)]=0
        else:
            # Explicit B2 minimum-support heuristic, not recovered author intent.
            weightxff.fill(0)
            weightxff[0,(index2[1]+index2[3])//2,(index2[0]+index2[2])//2]=1''')
    replace("np.where(weightxff.squeeze()>0.8)", "np.where(weightxff[0]>0.8)")
    replace('''        labelcls2[np.where(labelcls2<((w//2+h//2)*range/1.2)**2)]=1-((see-see.min())/(see.max()-see.min()+1e-4))
        weightcls3=np.zeros((1,size,size))
        weightcls3[np.where(labelcls2<((w//2+h//2)*range/1.2)**2)]=1
        labelcls2=labelcls2*weightcls3''', '''        if see.size:
            labelcls2[np.where(labelcls2<((w//2+h//2)*range/1.2)**2)]=1-((see-see.min())/(see.max()-see.min()+1e-4))
            weightcls3=np.zeros((1,size,size))
            weightcls3[np.where(labelcls2<((w//2+h//2)*range/1.2)**2)]=1
            labelcls2=labelcls2*weightcls3
        else:
            labelcls2.fill(0)
            labelcls2[0,(index[1]+index[3])//2,(index[0]+index[2])//2]=1''')
    replace('''           return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))''', '''           with np.errstate(over="ignore", invalid="ignore"):
               encoded=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
           invalid=~np.isfinite(encoded)
           if invalid.any():
               encoded[invalid]=np.tanh(x[invalid])
           return encoded''')
    replace("np.where(labelcls1.squeeze()==-1)", "np.where(labelcls1[0]==-1)")
    return source


if __name__ == "__main__":
    original = (ROOT / "third_party/HiFT" / RELATIVE).read_text(encoding="utf-8")
    diff = unified_diff(original.splitlines(True), corrected_source(original).splitlines(True),
                        fromfile="a/" + RELATIVE, tofile="b/" + RELATIVE, n=1)
    (ROOT / "patches/0005-empty-positive-support-b2.patch").write_text("".join(diff), encoding="utf-8", newline="\n")
