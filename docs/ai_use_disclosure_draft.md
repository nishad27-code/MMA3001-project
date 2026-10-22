# AI-use disclosure: factual draft

> **For the author.** Part A is a factual record of what the AI assistant did in the development session, based on the session itself. Part B must be written by you. The brief requires *your* reflection, and the assessment rewards evidence that you drove the process and understood the output. Do not keep any statement you cannot personally confirm.

## Part A: Factual record (AI assistant: Claude, Anthropic, via Claude Code)

**Tasks the AI performed, at the author's direction:**
1. Inspected the initial repository scaffold. Read the project brief and dataset PDFs and compared the three dataset options. It initially recommended Dataset 3 (images). The author chose Dataset 2 instead.
2. Ran read-only audits of the Dataset 2 files. It found that three of five environmental sensors contain only stale repeated packets, and that the occupancy-to-room mapping is marked "Incorrect Data" in the workbook. It recommended narrowing the scope to CO₂ gap reconstruction. The author approved the research question and scope.
3. Wrote all code in `src/gap_reconstruction/` and all tests in `tests/`. Also wrote `requirements.txt`, `pyproject.toml` and the `.gitignore` additions, set up the virtual environment, ran the extraction, experiments, tests and documentation generator, and inspected the figures.
4. Drafted `README.md` (keeping the author's existing AI Acknowledgement text), `docs/report_draft.md`, `docs/presentation_outline.md`, `docs/interview_prep.md` and this file.

**Errors and problems that occurred during AI-assisted work, and how they were found:**
- An early dataset audit scanned the 1.6 GB occupancy file in full and took too long. The author interrupted it twice and asked for quicker checks.
- The audit scripts initially failed twice:
  - Python 3.10's `datetime.fromisoformat` rejected the timestamp format;
  - a `KeyError` occurred because some JSON entries have no `value`.

  Both were fixed before any results were used.
- A unit test exposed a logic error in the AI's out-of-order counter (it compared with the previous reading instead of the latest time seen). The code was fixed and the test kept.
- The first timing measurements included harness overhead and varied widely between runs. Timing was changed to measure only the method call. The remaining run-to-run variation (up to about 4×) is reported rather than hidden.
- The first example-reconstruction plot drew curves from the first hidden reading rather than from the gap boundaries. This was corrected.
- Two example plots suggested a possible systematic bias. A bias metric and a check of the whole series showed no such effect.
- The report draft contained a garbled sentence, which was corrected.
- The acceptance reference (typical NDIR sensor accuracy, ventilation thresholds) was given without verified sources. It is flagged in the report for the author to source or remove.

**Decisions that stayed with the author:**
- choosing Dataset 2 over Dataset 3;
- approving the research question, the scope and the three methods;
- keeping raw data, processed data and reference PDFs out of Git;
- using a project-local virtual environment;
- requiring no commit or push without the author's review.

**Verification performed (by running code, not by assertion):**
- 98 pytest tests passed (`results/test-report.txt`).
- Extraction counts matched an independent audit script.
- Experiment metrics reproduced exactly on rerun.
- All report numbers were taken from the generated result files.

## Part B: Author's reflection [AUTHOR, to write]

- Approximate level of AI contribution, in your judgement, to code, analysis and writing.
- Why you used AI for those tasks.
- How *you* checked the AI-generated material (what you re-ran, read or compared against course notes).
- Which important decisions were yours (see the list above, and add your own).
- Errors, limitations or unhelpful suggestions you noticed yourself.
- How AI use affected your understanding and workflow, positively and negatively.
- What you can now explain without AI (methods, validation, results). You will need to do this in the interview.
