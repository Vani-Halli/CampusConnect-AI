# CampusConnect AI retrieval evaluation

- FAQ records loaded: 170
- Answerable test queries: 33
- Rejection test queries: 8
- Retrieval scores use the production retrieval engine and configured relevance thresholds.

## Overall results

| Mode | Status | Top-1 before -> after | Top-3 before -> after | MRR before -> after | Answerable accepted | Accepted top-1 | Rejection accuracy before -> after | False accepts before -> after |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tfidf | measured | 87.9% -> 87.9% | 93.9% -> 93.9% | 89.9% -> 90.4% | 97.0% | 90.6% | 12.5% -> 100.0% | 7 -> 0 |
| semantic | measured | 72.7% -> 87.9% | 90.9% -> 97.0% | 80.3% -> 92.4% | 100.0% | 87.9% | 12.5% -> 100.0% | 7 -> 0 |
| hybrid | measured | 87.9% -> 93.9% | 93.9% -> 97.0% | 90.9% -> 95.5% | 100.0% | 93.9% | 12.5% -> 100.0% | 7 -> 0 |

## Category results

### tfidf

| Category | Queries | Top-1 | Top-3 | MRR |
|---|---:|---:|---:|---:|
| Academic Verification | 3 | 66.7% | 66.7% | 66.7% |
| Administration | 3 | 100.0% | 100.0% | 100.0% |
| Admissions | 3 | 100.0% | 100.0% | 100.0% |
| College Contacts | 3 | 66.7% | 100.0% | 77.8% |
| Courses | 3 | 100.0% | 100.0% | 100.0% |
| Examinations | 3 | 66.7% | 100.0% | 83.3% |
| Fees | 3 | 100.0% | 100.0% | 100.0% |
| Placements | 3 | 100.0% | 100.0% | 100.0% |
| Scholarships | 3 | 100.0% | 100.0% | 100.0% |
| Student Life | 3 | 66.7% | 66.7% | 66.7% |
| Student Support | 3 | 100.0% | 100.0% | 100.0% |

### semantic

| Category | Queries | Top-1 | Top-3 | MRR |
|---|---:|---:|---:|---:|
| Academic Verification | 3 | 100.0% | 100.0% | 100.0% |
| Administration | 3 | 100.0% | 100.0% | 100.0% |
| Admissions | 3 | 100.0% | 100.0% | 100.0% |
| College Contacts | 3 | 66.7% | 100.0% | 83.3% |
| Courses | 3 | 100.0% | 100.0% | 100.0% |
| Examinations | 3 | 100.0% | 100.0% | 100.0% |
| Fees | 3 | 100.0% | 100.0% | 100.0% |
| Placements | 3 | 100.0% | 100.0% | 100.0% |
| Scholarships | 3 | 66.7% | 100.0% | 83.3% |
| Student Life | 3 | 33.3% | 66.7% | 50.0% |
| Student Support | 3 | 100.0% | 100.0% | 100.0% |

### hybrid

| Category | Queries | Top-1 | Top-3 | MRR |
|---|---:|---:|---:|---:|
| Academic Verification | 3 | 100.0% | 100.0% | 100.0% |
| Administration | 3 | 100.0% | 100.0% | 100.0% |
| Admissions | 3 | 100.0% | 100.0% | 100.0% |
| College Contacts | 3 | 66.7% | 100.0% | 83.3% |
| Courses | 3 | 100.0% | 100.0% | 100.0% |
| Examinations | 3 | 100.0% | 100.0% | 100.0% |
| Fees | 3 | 100.0% | 100.0% | 100.0% |
| Placements | 3 | 100.0% | 100.0% | 100.0% |
| Scholarships | 3 | 100.0% | 100.0% | 100.0% |
| Student Life | 3 | 66.7% | 66.7% | 66.7% |
| Student Support | 3 | 100.0% | 100.0% | 100.0% |

## Example retrievals (hybrid mode)

### Successful

- `paraphrase-bec001` (paraphrase, Admissions): What is BEC's full official name, and where is the college located? — expected `BEC001`, rank `1`, retrieved ['BEC001'].
- `paraphrase-bec008` (paraphrase, Courses): How many years does the B.E. course take at BEC? — expected `BEC008`, rank `1`, retrieved ['BEC008', 'BEC-ROW-111', 'BEC014'].
- `paraphrase-bec014` (paraphrase, Fees): Which components make up B.E. fees for 2026-27? — expected `BEC014`, rank `1`, retrieved ['BEC014', 'BEC021', 'BEC011'].

### Incorrect or low-ranked

- `paraphrase-bec072` (paraphrase, Student Life): Are student hostels available at BEC? — expected `BEC072`, rank `None`, retrieved ['BEC076', 'BEC077', 'BEC093'].
- `paraphrase-row153` (paraphrase, College Contacts): How can someone reach BEC for general enquiries? — expected `BEC-ROW-153`, rank `2`, retrieved ['BEC004', 'BEC-ROW-153', 'BEC078'].

### Rejection queries

- `unsupported-observatory`: Does BEC have a campus observatory that lends telescopes? — correctly rejected; retrieved [].
- `unsupported-aquarium`: What are the campus aquarium fish-feeding hours? — correctly rejected; retrieved [].
- `unsupported-telescope`: Where can I borrow a telescope from BEC? — correctly rejected; retrieved [].
- `unsupported-planetarium`: Can I reserve a planetarium ticket at BEC? — correctly rejected; retrieved [].
- `unsupported-observatory-hours`: What are the office hours for the campus observatory? — correctly rejected; retrieved [].
- `unsupported-drone`: Does BEC lend drones to students for weekend projects? — correctly rejected; retrieved [].
- `unsupported-podcast`: Does BEC run a public podcast recording studio? — correctly rejected; retrieved [].
- `empty-query`:  — correctly rejected; retrieved [].

## Interpretation and limitations

- This is a small, manually authored benchmark, not a representative sample of real student questions.
- Paraphrase quality and expected FAQ labels were authored by a person; results do not prove real-world accuracy.
- The original-question subset is naturally close to the indexed FAQ wording and should not be read as paraphrase performance.
- The rejection benchmark is small and manually authored, so its rejection accuracy is not a guarantee for unseen unsupported questions.
- The source FAQ records were not modified; missing source IDs use the loader's stable BEC-ROW-NNN IDs.
- Expected document IDs are consulted only after retrieval to score rankings; they are never passed into search.
