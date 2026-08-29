# Scientific Data manuscript draft

The manuscript is maintained in two synchronized sources:

- `main_vi.tex`: the approved Vietnamese content-review copy;
- `main_en.tex`: the English submission draft translated from that copy.

Both contain the full Scientific Data Data Descriptor structure:

- Title and author list
- Abstract
- Background & Summary
- Methods
- Data Records
- Technical Validation
- Data Availability
- Code Availability
- an embedded reference list
- Author Contributions, Competing Interests, and Funding

The Vietnamese copy remains the source of truth for resolving content questions.
Changes to numerical claims, methods, or release scope should be applied to both
language versions.

The manuscript deliberately uses the standard `article` class. Scientific Data
does not provide or recommend a LaTeX template, and revised LaTeX submissions
must not depend on a separate `.bib`, `.bbl`, or journal style file.

## Overleaf

Two independent Overleaf packages are generated:

- `dist/Scientific_Data_Vietnamese_Overleaf.zip`;
- `dist/Scientific_Data_English_Overleaf.zip`.

Each contains `main.tex`, all referenced figures, language-specific upload
instructions, provenance and submission checklists, and SHA-256 hashes. Upload
either ZIP as a new Overleaf project; no manual figure copying is needed. Figure
copies are retained under `figures/vi/` and `figures/en/`.

Rebuild the archive and copy the validated pipeline figures with:

```bash
python build_overleaf_packages.py --language vi
python build_overleaf_packages.py --language en
# or build both:
python build_overleaf_packages.py --language all
```

The archives contain the matching `SUBMISSION_CHECKLIST_<LANG>.md` and
`FIGURE_PROVENANCE_<LANG>.md`. Each contains four manuscript figures assembled
from eleven PNGs derived directly from the validated dataset pipeline. Figure 1
uses the same 15 pipeline-rendered terrain panels with language-specific shared
labels. The directory structure is kept as hand-written LaTeX `verbatim` text;
no generative-AI image is used.

The dataset repository and reserved DOI are fixed as Zenodo record
`10.5281/zenodo.22074838`. The code repository targets the immutable
`v1.0.0` GitHub release, and the dataset licence is fixed as CC BY 4.0. Author details and
the contribution, competing-interest, and funding declarations must also be
confirmed by the author.

## Submission-readiness checks

The existing data are sufficient to support the drafted Methods and Data
Records. Before treating the release and manuscript as submission-ready:

- publish the uploaded Zenodo dataset record reserved as
  `10.5281/zenodo.22074838`;
- preserve the exact source-code version used for the release under the
  `v1.0.0` Git tag and optionally archive it with a software DOI;
- retain the documented historical `3.1-preview` map-schema identifier for
  this immutable dataset release;
- retain the documented MacBook Pro M4, 16 GB, macOS 26.5.2 and three-worker
  execution context for current runtime values, or rerun timing with one worker
  if cross-machine single-worker latency is required;
- confirm the author's exact faculty/department wording;
- keep `main_en.tex` synchronized if any scientific content or release count is
  changed in `main_vi.tex`.
