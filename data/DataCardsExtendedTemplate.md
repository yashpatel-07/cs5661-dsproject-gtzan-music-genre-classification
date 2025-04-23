# GTZAN Dataset

The GTZAN Dataset is a widely used resource for music genre classification tasks. There are 1,000 audio tracks, each lasting 30 seconds. These tracks are spread evenly across 10 genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. Each track is a mono-channel WAV file sampled at 22,050 Hz.​





#### Dataset Link
<!-- info: Provide a link to the dataset: -->
<!-- width: half -->
[Dataset Link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)

The original site is no longer up, but can be found via [Internet Archive](https://web.archive.org/web/20200812034358/http://marsyas.info/downloads/datasets.html)

#### Data Card Author(s)
<!-- info: Select **one role per** Data Card Author:

(Usage Note: Select the most appropriate choice to describe the author's role
in creating the Data Card.) -->
<!-- width: half -->
- **CS5661 Group 2:** (Owner)

## Authorship
### Publishers
#### Publishing Organization(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the institution or organization responsible
for publishing the dataset: -->
Marsyas (Music Analysis, Retrieval and Synthesis for Audio Signals)

#### Industry Type(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** industry types to which the publishing
organizations belong: -->
- Academic - Tech
- Not-for-profit - Tech
- Individual - George Tzanetakis

#### Contact Detail(s)
<!-- scope: microscope -->
<!-- info: Provide publisher contact details: -->
- **Publishing POC:** George Tzanetakis - Main Designer and Developer
- **Affiliation:** University of Victoria - Department of Computer Science
- **Contact:** gtzan@cs.uvic.ca
- **Mailing List:** 
  - For [Users](https://sourceforge.net/projects/marsyas/lists/marsyas-users)
  - For [Developers](https://sourceforge.net/projects/marsyas/lists/marsyas-developers)
- **Website:**
  - Most recent Marsyas [archive link](https://web.archive.org/web/20220330025414/http://marsyas.info/downloads/datasets.html)
  - George Tzanetakis [school page](https://webhome.csc.uvic.ca/~gtzan/)

#### Authors (original 2002 publication in IEEE Transactions on Audio and Speech Processing)
<!-- scope: microscope -->
<!-- info: Provide the details of all authors associated with the dataset:

(Usage Note: Provide the affiliation and year if different from publishing
institutions or multiple affiliations.) -->
- George Tzanetakis, Student Member, IEEE, 2002
- Perry Cook, Member, IEEE, 2002

### Funding Sources
#### Institution(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the funding institution(s): -->
- NSF
- State of New Jersey Commission on Science and Technology
- Intel
- Ariel Foundation <!-- spelled as Arial in the paper, but probably a typo -->

#### Funding or Grant Summary(ies)
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide a short summary of programs or projects that may have funded
the creation, collection, or curation of the dataset.

Use additional notes to capture any other relevant information or
considerations. -->
- NSF under [Grant 9984087](https://www.nsf.gov/awardsearch/showAward?AWD_ID=9984087&HistoricalAwards=false)
- State of New Jersey Commission on Science and Technology under Grant 01-2042-007-22
- Intel
- Ariel Foundation

## Dataset Overview
#### Data Subject(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable**** subjects contained the dataset: -->
- Synthetically generated data
- Others - Music recorded from a variety of sources

#### Dataset Snapshot
<!-- scope: periscope -->
<!-- info: Provide a snapshot of the dataset:<br><br>(Use the additional notes
to include relevant information, considerations, and links to table(s) with
more detailed breakdowns.) -->
Category | Data
--- | ---
Size of Dataset | 1.2 GB
Number of Instances | 1000
Length of Instance | 30 seconds
Labeled Classes | 10
Number of Labels | 1000
Average Labels Per Instance | 1
Average Instances Per Label | 100
Frequency | 22050 MHz
Channel | Mono
Bit Depth | 16
Format | .wav

**Above:** Dataset snapshot based off the audio data

#### Content Description
<!-- scope: microscope -->
<!-- info: Provide a short description of the content in a data point: -->
The dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

#### Descriptive Statistics
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each field.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Some statistics will be relevant for numeric data, for not for
strings. -->

<!-- Statistics taken from features_30_sec.csv from the Kaggle set, might have to change as their jazz #54 was broken.-->
Statistic | length (bytes) |
--- | --- | 
count | 1000 | 
mean | 662030.846000 | 
std | 1784.073992 | 
min | 660000 |
25% | 661504 |
50% | 661794 |
75% | 661794 |
max | 661794 |
mode | 675808 |

**Above:** Length (bytes) of wav files description

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

**Last Updated:** 2002

**Release Date:** 2002

#### Maintenance Plan
<!-- scope: microscope -->
<!-- info: Summarize the maintenance plan for the dataset:

Use additional notes to capture any other relevant information or
considerations. -->
It's a very famous dataset that has been regularly used as a benchmark for music genre classification for over two decades now.

**Errors and Feedback:** [https://arxiv.org/abs/1306.1461]() goes into depth of its faults

## Example of Data Points
#### Primary Data Modality
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Audio Data

#### Sampling of Data Points
<!-- scope: periscope -->
<!-- info: Provide link(s) to data points or exploratory demos: -->
- [https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data]()

#### Typical Data Point
<!-- width: half -->
<!-- info: Provide an example of a typical data point and describe what makes
it typical.

**Use additional notes to capture any other relevant information or
considerations.** -->
Summarize here. Include any criteria for typicality of data point.

```
>>> file = wave.open("/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/pop/pop.00032.wav")
>>> file.getparams()
_wave_params(nchannels=1, sampwidth=2, framerate=22050, nframes=661504, comptype='NONE', compname='not compressed')
```

## Motivations & Intentions
### Motivations
#### Purpose(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Educational Attainment

#### Domain(s) of Application
<!-- scope: periscope -->
<!-- info: Provide a list of key domains of application that the dataset has
been designed for:<br><br>(Usage Note: Use comma-separated keywords.) -->

`Machine Learning`, `Music Genre Classification`, `Deep Learning`

#### Motivating Factor(s)
<!-- scope: microscope -->
<!-- info: List the primary motivations for creating or curating this dataset:

(Usage Note: use this to describe the problem space and corresponding
motivations for the dataset.) -->
- Explore a well-understood and studied dataset to learn how to create models to classify musical genres.

### Intended Use
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe for production use

#### Citation Guidelines
<!-- scope: microscope -->
<!-- info: Provide guidelines and steps for citing this dataset in research
and/or production.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Guidelines & Steps:** 

**BiBTeX:**
```
@ARTICLE{1021072,
  author={Tzanetakis, G. and Cook, P.},
  journal={IEEE Transactions on Speech and Audio Processing}, 
  title={Musical genre classification of audio signals}, 
  year={2002},
  volume={10},
  number={5},
  pages={293-302},
  keywords={Humans;Music information retrieval;Instruments;Computer science;Multiple signal classification;Signal analysis;Pattern recognition;Feature extraction;Wavelet analysis;Cultural differences},
  doi={10.1109/TSA.2002.800560}}
```

## Provenance
### Collection
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data: -->
- Taken from mediums of recording

#### Methodology Detail(s)
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for collection method
type.) -->
**Collection Type**

**Source:** Collected by George Tzanetakis in an undocumented fashion.

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** 2000-2001

**Primary modality of collection data:**

- Audio Data

**Update Frequency for collected data:**

- Unknown

## Human and Other Sensitive Attributes
#### Sensitive Human Attribute(s)
<!-- scope: telescope -->
<!-- info: Select **all attributes** that are represented (directly or
indirectly) in the dataset. -->
- Gender
- Socio-economic status
- Geography
- Language
- Culture
- Experience or Seniority

#### Rationale
<!-- scope: microscope -->
<!-- info: Describe the motivation, rationale, considerations or approaches
that caused this dataset to include the indicated human attributes.

Summarize why or how this might affect the use of the dataset. -->
Music is an art created by humans. The recordings taking place between 2000-2001 by a Ph.D. CS graduate student shows the type of music and mediums they had access to in their space and time.

## Extended Use
### Use with Other Data
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to use with other data

### Forking & Sampling
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to form and/or sample

#### Acceptable Sampling Method(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** acceptable methods to sample this
dataset: -->
- Cluster Sampling
- Haphazard Sampling
- Multi-stage sampling
- Random Sampling
- Retrospective Sampling
- Stratified Sampling
- Systematic Sampling
- Weighted Sampling
- Unknown
- Unsampled
- Others

### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
- Training
- Testing
- Validation
- Development or Production Use
- Fine Tuning
- Others

## Sampling Methods
<!-- info: Fill out the following block if your dataset employs any sampling
methods. -->
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used in the creation of this
dataset: -->
- Random Sampling
- Unknown

## Known Applications & Benchmarks
<!-- info: Fill out the following section if your dataset was primarily
created for use in AI or ML system(s) -->
#### ML Application(s)
<!-- scope: telescope -->
<!-- info: Provide a list of key ML tasks
that the dataset has been
used for.

Usage Note: Use comma-separated keywords. -->
`Classification`
