import ot
import tifffile
import itertools
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from data_analysis_scripts.utils import tk_popover, minmax_norm

fnames = tk_popover(open_many=True)
imgs = [minmax_norm(tifffile.imread(fname)) for fname in fnames]

# %%
fig, axs = plt.subplots(4, 5)
for i, ax in enumerate(axs.flat):
    ax.imshow(imgs[i], cmap="bone")
    ax.title.set_text(i)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
pairs = itertools.combinations(list(range(len(imgs))), 2)
pairwise_emds = [ot.sliced_wasserstein_distance(imgs[a], imgs[b], n_projections=1000) for a, b in pairs]

# %%
plt.hist(pairwise_emds, bins=20)
plt.show()

# %%
shapirotest = shapiro(pairwise_emds)
if shapirotest.pvalue < 0.05:
    t = "not"
else:
    t = ""
print(f"Distribution is {t} normally distributed.")

#%%
