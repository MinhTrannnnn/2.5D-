# Scientific Data submission checklist

## Present in this project

- 5,010 JSON map files, 50,100 map--task records, and 90,180 benchmark trials.
- README, schema, example reader, version-pinned environment, and regression tests.
- Machine-readable validation report, file inventory, and SHA-256 checksums.
- A Data Descriptor manuscript containing all required non-optional headings.
- A reference list embedded in the single `.tex` source.
- Sixteen verified literature references covering FAIR principles, software,
  planner origins, and eight closely related off-road datasets/navigation
  benchmarks, plus the formal Zenodo data citation.
- A scoped comparison table distinguishes acquisition modality, release scale,
  traversability conditioning, paired difficulty, planning artifacts, and
  public release support without treating perception frames as navigation maps.
- Methods now state the complete family distributions, all detail-field
  parameters, the exact 34-value severity grid, acceptance metrics and score,
  digital footprint and boundary rule, deterministic task fallbacks, and the
  end-to-end matched-generation algorithm.
- Four manuscript figures assembled from eleven genuine pipeline or benchmark
  PNG files; no GenAI-generated image is used.
- The dataset licence is fixed as CC BY 4.0; the software uses the MIT License.
- The historical `3.1-preview` map-schema label is intentionally retained and
  documented for this immutable dataset release.
- Release-generation configuration documented in Methods: MacBook Pro
  (Mac16,1), Apple M4 10-core CPU, 16-GB unified memory, macOS 26.5.2, up to six
  generation workers, and three benchmark workers.

## Required before initial submission

- Publish the uploaded Zenodo record reserved as
  `10.5281/zenodo.22074838`; the repository, DOI, and formal data citation are
  already present in both manuscript sources.
- Push the prepared `v1.0.0` code tag and optionally archive it with a software
  DOI before submission.
- Confirm the author's affiliation wording, name, ORCID, and correspondence
  address.
- Confirm the single-author contribution statement. Funding currently states
  that no external funding was received. Competing Interests currently states
  that none exist; change it if a financial, professional, or personal
  relationship could reasonably be perceived to influence the work.

## At submission

- The initial submission may use a main PDF with figures and tables embedded,
  but the data must be accessible through an anonymous URL or formal repository.
- A revised LaTeX submission should use one self-contained `.tex` file without
  a separate `.bib`, `.bbl`, or journal style; figures must also be uploaded as
  separate files when requested by the submission system.
- The cover letter is a required administrative file, but data-access
  instructions must appear in the manuscript rather than only in the letter.
- Replace every bold bracketed placeholder in `main.tex` before submission.
