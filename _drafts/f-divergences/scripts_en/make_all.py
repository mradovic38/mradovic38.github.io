"""Regenerate every blog figure: python scripts_en/make_all.py (needs numpy, matplotlib, ffmpeg)."""

import bayes_risk
import empirical_cdf
import forward_reverse_kl
import generators
import kl_estimators
import le_cam
import local_fisher
import ot_vs_mixture
import transport
import tv_area
import wasserstein

for module in (empirical_cdf, generators, forward_reverse_kl, tv_area, bayes_risk, le_cam,
               local_fisher, kl_estimators, transport, wasserstein, ot_vs_mixture):
    print(f"--- {module.__name__}")
    module.main()
