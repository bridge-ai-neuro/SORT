# BookSORT
BookSORT is a dataset created from books for evaluation on the Sequence Order Recall Task (SORT), w hich assesses a model's ability to use temporal context in memory. SORT evaluation samples can be constructed from any sequential data. For BookSORT, the sequences are derived from text from 9 English language books that were released to the public domain between 2022 and 2024 via Project Gutenberg.

SORT presents models with two segments of text from a continuous sequence, like text, and asks the model to judge the order in which they appeared. In one SORT condition, the relevant text excerpt is provided as additional context to the model to help it perform the task. This BookSORT dataset varies text excerpt lengths, segment pair lengths, and distances between segment pairs.

#### Dataset Link
<!-- info: Provide a link to the dataset: -->
<!-- width: half -->
[TODO Dataset Link](TODOLINK)

<!--- FILL IN AFTER SUBMISSION -->
#### Data Card Author(s)
<!-- info: Select **one role per** Data Card Author:
(Usage Note: Select the most appropriate choice to describe the author's role
in creating the Data Card.) -->
<!-- width: half -->
- **Name, Team:** (Owner / Contributor / Manager)

## Authorship
### Publishers
#### Publishing Organization(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the institution or organization responsible for publishing the dataset: -->
Organization Name

#### Industry Type(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** industry types to which the publishing organizations belong: -->
<!--- FILL IN AFTER SUBMISSION
- Corporate - Tech
- Corporate - Non-Tech (please specify)
- Academic - Tech
- Academic - Non-Tech (please specify)
- Not-for-profit - Tech
- Not-for-profit - Non-Tech (please specify)
- Individual (please specify)
- Others (please specify)
-->

#### Contact Detail(s)
<!-- scope: microscope -->
<!-- info: Provide publisher contact details: -->
<!--- FILL IN AFTER SUBMISSION
- **Publishing POC:** Provide the name for a POC for this dataset's publishers
- **Affiliation:** Provide the POC's institutional affiliation
- **Contact:** Provide the POC's contact details
- **Mailing List:** Provide a mailing list if available
- **Website:** Provide a website for the dataset if available
-->

### Dataset Owners
#### Team(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the groups or team(s) that own the dataset: -->
Name of Group or Team

#### Contact Detail(s)
<!-- scope: periscope -->
<!-- info: Provide pathways to contact dataset owners: -->
<!--- FILL IN AFTER SUBMISSION
- **Dataset Owner(s):** Provide the names of the dataset owners
- **Affiliation:** Provide the affiliation of the dataset owners
- **Contact:** Provide the email of the dataset owner
- **Group Email:** Provide a link to the mailing-list@server.com for the dataset owner team
- **Website:** Provide a link to the website for the dataset owner team
-->

#### Author(s)
<!-- scope: microscope -->
<!-- info: Provide the details of all authors associated with the dataset:
(Usage Note: Provide the affiliation and year if different from publishing
institutions or multiple affiliations.) -->
- Name, Title, Affiliation, YYYY

### Funding Sources
#### Institution(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the funding institution(s): -->
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

## Dataset Overview

The dataset consists of text samples and metadata from 9 public domain books from Project Gutenberg.

To evaluate text on the Sequence Order Recall Task (SORT), we extracted text excerpts $E$ and pairs of text segments $S$ contained within those excerpts. As detailed in the accompanying paper, BookSORT varied the length of the text excerpts $L_E$, the length of the segments $L_S$, and the distance between the segments $D_S$. All excerpts and segments began at a sentence boundary, and units of length and distance are computed in words.

Each unique combination of excerpt lengths $L_E$ and segment lengths $L_S$ produced 3 `.csv` files containing (1) information about the included books, (2) information about the excerpts from those books, and (3) information about the segments from those excerpts.

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
| Extended Context Length | $L_S$   | $1000$    | $L_E / 4$ | $L_E / 2$ | $L_E / 0.8$ |

**Above:** The definition of the segment distance bins that determine how far apart the text segments are from one another. Distance is defined by the beginning of the first segment to the beginning of the second segment.

We only evaluated the Sequence Order Recall Task on the first 100 segment pairs in each combination of book, $L_E$, $L_S$, and $L_D$. The remaining pairs are reserved for other uses (e.g. selecting which prompt format produces the best SORT results).

A complete description of the data fields is given in the [BookSORT metadata](booksort/BookSORT_metadata.json) following the MLCroissant 1.0 specification.

#### Data Subject(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable**** subjects contained the dataset: -->

The content of the books contains the following:

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
  author={Last, First and Last, First and others},
  conference={placeholder},
  year={2020},
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
- [GitHub URL](data/booksort)

## Provenance
### Collection
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data: -->
- API

#### Methodology Detail(s)
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for collection method
type.) -->
**Collection Type**

**Source:** The full text of the books are taken from [https://gutenberg.org/](https://gutenberg.org/). This follows their [license](https://www.gutenberg.org/policy/license.html) specifying the terms of use. 

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** [01 2022 - 03 2024]

**Primary modality of collection data:** Text Data

**Update Frequency for collected data:** Static

#### Source Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of each upstream source of data.

Use additional notes to capture any other relevant information or
considerations. -->

| ID    | Title                              | Author                             | Word count | Release   | Pub  | LoCC | Subjects                                                                                                            |
|-------|------------------------------------|------------------------------------|------------|-----------|------|----------------------|---------------------------------------------------------------------------------------------------------------------|
| 69087 | The Murder of Roger Ackroyd        | Christie, Agatha                   | 69,720     | 10/2/2022 | 1926 | PR                   | Detective and mystery stories; Fiction: Private investigators - England, Murder - Investigation, Belgians - England |
| 72578 | Tom Swift and His Talking Pictures | Appleton, Victor                   | 43,853     | 1/1/2024  | 1928 | PZ                   | Adventure stories; Motion pictures                                                                                  |
| 72600 | The Trumpeter of Krakow            | Kelly, Eric Philbrook              | 59,081     | 1/2/2024  | 1928 | PZ                   | Juvenile fiction: Middle Ages, Poland - History - Casimir IV, 1447-1492                                             |
| 72869 | Meet the Tiger                     | Charteris, Leslie                  | 79,946     | 2/4/2024  | 1928 | PR                   | Fiction: Private investigators - England; Detective and mystery stories                                             |
| 72958 | Hunting for Hidden Gold            | Dixon, Franklin W.                 | 42,354     | 2/14/2024 | 1928 | PZ                   | Juvenile fiction: Brothers, Gold mines and mining, Montana, Robbers and outlaws; Mystery and detective stories      |
| 72963 | The Nature of the Physical World   | Eddington, Arthur Stanley, Sir     | 104,530    | 2/15/2024 | 1928 | Q                    | Physics - Philosophy; Science - Philosophy                                                                          |
| 72972 | Money for Nothing                  | Wodehouse, P.G. (Pelham Grenville) | 82,331     | 2/16/2024 | 1928 | PR                   | Humorous stories; Fiction: Swindlers and swindling, Greed                                                           |
| 73017 | Pomona; or, the Future of English  | De Selincourt, Basil               | 9,273      | 2/22/2024 | 1928 | PE                   | English language                                                                                                    |
| 73042 | The Well of Loneliness             | Hall, Radclyffe                    | 163,217    | 2/26/2024 | 1928 | PR                   | Fiction: Lesbians - England - Social conditions                                                                     |

**Above:** Project Gutenberg metadata for the books in this dataset.

#### Data Processing
<!-- scope: microscope -->
<!-- info: Summarize how data from different sources or methods aggregated,
processed, or connected.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the following for each source OR
collection method.) -->
**Preprocessing text**

**Description:** We wrote custom Python code to only retain the book text that formed a continuous narrative. We stripped the front and back matter of the book, and extracted chapter titles if they existed. 8 of the 9 books contained individual section or chapter breaks. For these 8 books, we parsed the text corresponding to each chapter. Chapter titles or section headings (e.g. 'VI' to indicate section six) were removed, and all remaining text was concatenated. This string was split into words (assuming simple whitespace separators with python `string.split()`) to produce a final text array for each book. This text array was sampled for the BookSORT dataset.

### Collection Criteria
#### Data Selection
<!-- scope: telescope -->
<!-- info: Summarize the data selection criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **Book selection:** We followed Project Gutenberg [guidelines](https://gutenberg.org/policy/robot_access.html) for crawling the site. First we downloaded a catalog of book metadata. We filtered this metadata to only view books released in 2024, and originally published in 1928 (thus passing the 95 year mark for copyright to expire). Titles were manually selected to attempt to maximize diversity over the Library of Congress Classification (LoCC), and to have some range in subject matter and book length. These filtered titles were then examined to check that they contained a continuous narrative across the entire book (i.e. not collections of stories or poems), and were therefore appropriate for the SORT evaluation.  

### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
- Testing
- Validation

## Sampling Methods
<!-- info: Fill out the following block if your dataset employs any sampling
methods. -->
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used in the creation of this
dataset: -->
- Multi-stage Sampling
- Random Sampling

The text excerpts and segments are all sampled from randomly and uniformly from across the text. Since we required all of these to begin at a sentence boundary, we first found all the relevant sentence boundaries and sampled uniformly from this set. Specific details can be found in the release of the dataset creation code [here](../sort/dataset_creation/).
 
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
This report evaluated several state-of-the-art LLMs across different families: Mistral, Mixtral (Mixture of Experts models), Llama-2, Llama-3, Gemma, and OpenAI GPT models.

<!---
## Terms of Art
### Concepts and Definitions referenced in this Data Card
<!-- info: Use this space to include the expansions and definitions of any
acronyms, concepts, or terms of art used across the Data Card.
Use standard definitions where possible. Include the source of the definition
where indicated. If you are using an interpretation,
adaptation, or modification of the standard definition for the purposes of your
Data Card or dataset, include your interpretation as well. -->
<!---
#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here

#### Term of Art
Definition: Write here

Source: Write here and share link

Interpretation: Write here
--> -->
