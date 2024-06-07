# BookSORT
BookSORT is a dataset created from books for evaluation on the Sequence Order Recall Task (SORT), w hich assesses a model's ability to use temporal context in memory. SORT evaluation samples can be constructed from any sequential data. For BookSORT, the sequences are derived from text from 9 English language books that were released to the public domain between 2022 and 2024 via Project Gutenberg.

SORT presents models with two segments of text from a continuous sequence, like text, and asks the model to judge the order in which they appeared. In one SORT condition, the relevant text excerpt is provided as additional context to the model to help it perform the task. This BookSORT dataset varies text excerpt lengths, segment pair lengths, and distances between segment pairs.

#### Dataset Link
<!-- info: Provide a link to the dataset: -->
<!-- width: half -->
Dataset Link

#### Data Card Author(s)
<!-- info: Select **one role per** Data Card Author:

(Usage Note: Select the most appropriate choice to describe the author's role
in creating the Data Card.) -->
<!-- width: half -->
- **Name, Team:** (Owner / Contributor / Manager)
- **Name, Team:** (Owner / Contributor / Manager)
- **Name, Team:** (Owner / Contributor / Manager)

## Authorship
### Publishers
#### Publishing Organization(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the institution or organization responsible
for publishing the dataset: -->
Organization Name

#### Industry Type(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** industry types to which the publishing
organizations belong: -->
- Corporate - Tech
- Corporate - Non-Tech (please specify)
- Academic - Tech
- Academic - Non-Tech (please specify)
- Not-for-profit - Tech
- Not-for-profit - Non-Tech (please specify)
- Individual (please specify)
- Others (please specify)

#### Contact Detail(s)
<!-- scope: microscope -->
<!-- info: Provide publisher contact details: -->
- **Publishing POC:** Provide the name for a POC for this dataset's publishers
- **Affiliation:** Provide the POC's institutional affiliation
- **Contact:** Provide the POC's contact details
- **Mailing List:** Provide a mailing list if available
- **Website:** Provide a website for the dataset if available

### Dataset Owners
#### Team(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the groups or team(s) that own the dataset: -->
Name of Group or Team

#### Contact Detail(s)
<!-- scope: periscope -->
<!-- info: Provide pathways to contact dataset owners: -->
- **Dataset Owner(s):** Provide the names of the dataset owners
- **Affiliation:** Provide the affiliation of the dataset owners
- **Contact:** Provide the email of the dataset owner
- **Group Email:** Provide a link to the mailing-list@server.com for the dataset owner team
- **Website:** Provide a link to the website for the dataset owner team

#### Author(s)
<!-- scope: microscope -->
<!-- info: Provide the details of all authors associated with the dataset:

(Usage Note: Provide the affiliation and year if different from publishing
institutions or multiple affiliations.) -->
- Name, Title, Affiliation, YYYY
- Name, Title, Affiliation, YYYY
- Name, Title, Affiliation, YYYY
- Name, Title, Affiliation, YYYY

### Funding Sources
#### Institution(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the funding institution(s): -->
- Name of Institution
- Name of Institution
- Name of Institution

#### Funding or Grant Summary(ies)
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide a short summary of programs or projects that may have funded
the creation, collection, or curation of the dataset.

Use additional notes to capture any other relevant information or
considerations. -->
*For example, Institution 1 and institution 2 jointly funded this dataset as a
part of the XYZ data program, funded by XYZ grant awarded by institution 3 for
the years YYYY-YYYY.*

Summarize here. Link to documents if available.

**Additional Notes:** Add here

## Dataset Overview

The dataset consists of text samples and metadata from 9 public domain books from Project Gutenberg.

To evaluate text on the Sequence Order Recall Task (SORT), we extracted text excerpts $E$ and pairs of text segments $S$ contained within those excerpts. As detailed in the accompanying paper, BookSORT varied the length of the text excerpts $L_E$, the length of the segments $L_S$, and the distance between the segments $D_S$. All units of length and distance are computed in words. Each unique combination of excerpt lengths $L_E$ and segment lengths $L_S$ produced 3 `.csv` files containing (1) information about the included books, (2) information about the excerpts from those books, and (3) information about the segments from those excerpts.

Since we evaluated LLMs with varying maximum context windows, we constructed a dataset for fairly standard context length limits (providing text excerpts up to 2500 words to fit within 4096 tokens) and for extended context length limits (providing 10K-20K word excerpts).

#### Dataset Snapshot
<!-- scope: periscope -->
<!-- info: Provide a snapshot of the dataset:<br><br>(Use the additional notes
to include relevant information, considerations, and links to table(s) with
more detailed breakdowns.) -->

Category | Data
--- | ---
Size of Dataset | 341 MB
Number of Instances | 37850
Number of Fields | 13

**Above:** Summary statistics about the BookSORT dataset. 

#### Dataset Details
We created data samples for 5 different excerpt lengths ($L_E=\{250, 1000, 2500, 10000, 20000\}$ words) and 2 segment lengths ($L_S=\{20, 50\}$ words). For each unique combination of $L_E$ and $L_S$, we sampled 110 excerpts from each included book. Most of the dataset used all 9 books; 1 book is excluded from the extended excerpt length data as it is shorter than 10000 words.

Within each unique book excerpt, we sampled segment pairs with varying distances between them. 110 segment pairs were sampled for 4 different distance bins, yielding 440 SORT trials per book, excerpt length, and segment length. Since distance is bounded by the excerpt length, we generally used $L_E$ to scale the bin edges.

| Condition               | Minimum | Bin0      | Bin1      | Bin2      | Bin3        |
|-------------------------|---------|-----------|-----------|-----------|-------------|
| Standard Context Length | $L_S$   | $L_E / 4$ | $L_E / 3$ | $L_E / 2$ | $L_E / 0.8$ |
| Extended Context Length | $L_S$   | 1000      | $L_E / 4$ | $L_E / 2$ | $L_E / 0.8$ |

**Above:** The definition of the segment distance bins that determine how far apart the text segments are from one another. 

A complete description of the data fields is given in the BookSORT metadata following the MLCroissant 1.0 specification. [TODOLINK](TODOLINK)

[TODO] Include Table of the book metadata

**Above:** Information about the books included in the dataset. 

#### Data Subject(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable**** subjects contained the dataset: -->
- Non-Sensitive Data about people
- Data about natural phenomena
- Data about places and objects

### Sensitivity of Data
#### Sensitivity Type(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable*** data types present in the dataset: -->
- None

#### Risk Type(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** risk types presenting from the
dataset: -->
- No Known Risks

### Dataset Version and Maintenance
#### Maintenance Status
<!-- scope: telescope -->
<!-- info: Select **one:** -->
**Limited Maintenance** - The data will not be updated,
but any technical issues will be
addressed.

#### Version Details
<!-- scope: periscope -->
<!-- info: Provide details about **this** version of the dataset: -->
**Current Version:** 1.0

**Last Updated:** 05/2024

**Release Date:** 06/2024

#### Maintenance Plan
<!-- scope: microscope -->
<!-- info: Summarize the maintenance plan for the dataset:

Use additional notes to capture any other relevant information or
considerations. -->
We do not anticipate continued updates of the BookSORT dataset. As SORT datasets can be programatically constructed from any sequential data, users may pull the original dataset creation source code and create different versions of the dataset with other book text or other sequential data as they see fit. Any new versions or updates will only be released in case we discover technical errors.

## Motivations & Intentions
### Motivations
#### Purpose(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Research

#### Domain(s) of Application
<!-- scope: periscope -->
<!-- info: Provide a list of key domains of application that the dataset has
been designed for:<br><br>(Usage Note: Use comma-separated keywords.) -->
`Machine Learning`, `Natural Language Processing`, `Deep Learning`

#### Motivating Factor(s)
<!-- scope: microscope -->
<!-- info: List the primary motivations for creating or curating this dataset:

(Usage Note: use this to describe the problem space and corresponding
motivations for the dataset.) -->
BookSORT was created to accompany the paper introducing the Sequence Order Recall Task (SORT). The primary motivation was to evaluate several state-of-the-art LLMs on the task. It is shared for reproducibility purposes.

### Intended Use
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe for research use

#### Citation Guidelines
<!-- scope: microscope -->
<!-- info: Provide guidelines and steps for citing this dataset in research
and/or production.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Guidelines & Steps:** While we release this dataset under CC0, please consider citing the accompanying paper if you use this dataset or any derivative of it.

**BiBTeX:**
```
@article{placeholder,
  title={placeholder},
  author={Kuznetsova, Alina and Rom, Hassan and Alldrin, and others},
  journal={International Journal of Computer Vision},
  volume={128},
  number={7},
  pages={1956--1981},
  year={2020},
  publisher={Springer}
}
```

## Access, Retention, & Wipeout
### Access
#### Access Type
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- External - Open Access

#### Documentation Link(s)
<!-- scope: periscope -->
<!-- info: Provide links that describe documentation to access this
dataset: -->
- [Dataset Website URL](TODO)
- [GitHub URL](TODO)

## Provenance
### Collection
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data: -->
- API
- Taken from other existing datasets

#### Methodology Detail(s)
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for collection method
type.) -->
**Collection Type**

**Source:** Describe here. Include links where available.

**Platform:** [Platform Name], Describe platform here. Include links where relevant.

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** [MMM YYYY - MMM YYYY]

**Primary modality of collection data:** Text Data

**Update Frequency for collected data:** Static

#### Source Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of each upstream source of data.

Use additional notes to capture any other relevant information or
considerations. -->
- **Source:** Describe here. Include links, data examples, metrics, visualizations where relevant.

#### Data Processing
<!-- scope: microscope -->
<!-- info: Summarize how data from different sources or methods aggregated,
processed, or connected.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the following for each source OR
collection method.) -->
**Collection Method or Source**

**Description:** Describe here. Include links where relevant.

**Methods employed:** Describe here. Include links where relevant.

**Tools or libraries:** Describe here. Include links where relevant.

**Additional Notes:** Add here

### Collection Criteria
#### Data Selection
<!-- scope: telescope -->
<!-- info: Summarize the data selection criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Collection Method of Source:** Summarize data selection criteria here. Include links where available.

**Additional Notes:** Add here

#### Data Inclusion
<!-- scope: periscope -->
<!-- info: Summarize the data inclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Collection Method of Source:** Summarize data inclusion criteria here. Include links where available.

**Additional Notes:** Add here

### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
- Testing
- Validation

#### Distribution(s)
<!-- scope: periscope -->
<!-- info: Describe the recommended splits and corresponding criteria.

Use additional notes to capture any other
relevant information or considerations. -->

Set | Number of data points
--- | ---
Train | 62,563
Test | 62,563
Validation | 62,563
Dev | 62,563

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Library(ies) and Method(s) Used
<!-- scope: microscope -->
<!-- info: Provide a description of the methods
used to transform or process the
dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied.) -->
**Transformation Type**

**Method:** Describe the transformation
method here. Include links where
necessary.

**Platforms, tools, or libraries:**
- Project Gutenberg metadata: Write description here
- Platform, tool, or library: Write description here

**Transformation Results:** Full text, arrays with chapter titles stripped 

### Breakdown of Transformations
<!-- info: Fill out relevant rows. -->

## Sampling Methods
<!-- info: Fill out the following block if your dataset employs any sampling
methods. -->
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used in the creation of this
dataset: -->
- Multi-stage Sampling
- Random Sampling
- Stratified Sampling

#### Characteristic(s)
<!-- scope: periscope -->
<!-- info: Provide characteristics of each sampling
method used.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each sampling method
used.) -->
**(Sampling Type)** | **Number**
--- | ---
Upstream Source | Write here
Total data sampled | 123m
Sample size | 123
Threshold applied | 123k units at property
Sampling rate | 123
Sample mean | 123
Sample std. dev | 123
Sampling distribution | 123
Sampling variation | 123
Sample statistic | 123

**Above:** Provide a caption for the above table or visualization.

**Additional Notes:** Add here

#### Sampling Criteria
<!-- scope: microscope -->
<!-- info: Describe the criteria used to sample data from
upstream sources.

Use additional notes to capture any other
relevant information or considerations. -->
- **Sampling method:** Summarize here. Include links where applicable.

## Known Applications & Benchmarks
<!-- info: Fill out the following section if your dataset was primarily
created for use in AI or ML system(s) -->

#### Evaluation Result(s)
<!-- scope: periscope -->
<!-- info: Provide the evaluation results from
models that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
A thorough report on evaluation can be found in the original accompanying [paper](TODOLINK).
We evaluated [TODO] MODEL-LIST.

## Terms of Art
### Concepts and Definitions referenced in this Data Card
<!-- info: Use this space to include the expansions and definitions of any
acronyms, concepts, or terms of art used across the Data Card.
Use standard definitions where possible. Include the source of the definition
where indicated. If you are using an interpretation,
adaptation, or modification of the standard definition for the purposes of your
Data Card or dataset, include your interpretation as well. -->
#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

