## Foreward

<p style="line-height:2">

The test predictions are just for a subset of the bird species, 20 or so and that's all I'll be investigating for now. But 
eventually I'll build up the base here as an input into the actual model, and train on all the species. 
</p>

<p style="line-height:2">

I honestly have no background in sound processing, or the technique I'm using [Matrix Profiling](https://stumpy.readthedocs.io/en/latest/index.html)
But I think I can viably do a basic example of a small subset of the training data and submit at least one kaggle
prediction set, which will likely be terrible since the entire training data is 6 GB and computationally this would take too long to predict.
</p>

<p style="line-height:2">

I chose this since I nearly asked to switch my capstone to the topic of testing this technique in sound processing, but I decided a full analysis
would not be possible. So far all I've done is watch alot of videos about the topics but I've been eager to try something out. 
</p>

<p style="line-height:2">

However I definately intend to continue work and predict using all of the training data after the course, and write a blog about
the applicability of this technique in sound processing at scale. 
</p>


<p style="line-height:2">

Another added benefit as since, from my research, this a relatively new method in python, not many submissions will utilize it. and for
that reason I forsee it being an epic fail, or a potential silver bullet - but odds on the former. 
</p>


**UPDATE**

- I wrote this yesterday when I started but I definately put much more time in than a hackathon and this point I'm not able to complete submission to kaggle.
- I did however get a good idea of this approaches feasibility going forward. 
- The EDA notebook contains all the work, and the submission is the format for submissions, where you insert your model into a part of the code. 

# Introduction

<p style="line-height:2">

As part of a 1 day hackathon I'm attemping to build a basic model and submit one set of predictions for the following
[kagle competition](https://www.kaggle.com/c/birdclef-2022)

Which aims to classify bird species via bioacoustic monitoring, for monitoring endangered species.
</p>

# Background


<p style="line-height:2">

My first introduction to sound processing was a few weeks ago, via this sci-py [presentation](https://www.youtube.com/watch?v=0ALKGR0I5MA)

I've wanted to tackle a sound processing classification challenge since I discovered the concept of 'matrix profiling', as part of researching time series methods for an energy demand forecasting project. This [presentation](https://www.youtube.com/watch?v=T9_z7EpA8QM) gives an overview of matrix profiling, and it's new implementation in python via 'stumpy'.

Matrix profiling has been around for quite some time, but has thus far been computationaly to expensive for modern classification challenges. A more efficient method of matrix profiling was recently developed by researchers at the University of California, and with those findings a python library was developed.

In their presentation slides, the researchers offer several use cases for the technique. One being sound processing, and I have been curious as to whether matrix profiling would be more efficient than the traditional FFT transform and Supervised Learning.

So far I haven't come across anyone using matrix profiling for a sound classification challenge, and the examples I've seen have been comparisons of one song to another, rather than multiple comparisons and making a conclusion with the actual matrix profile values.
</p>


<p style="line-height:2">

My process and theory is as follows, and the applicability of such is contingent on runtime:

Since matrix-profiling is resource intensive, it might not be relevant for sound classification. My approach would be to take an untransformed wave (the un-differentiated mix of different frequency information) and treat this wave as a time series trend. The sound byte would then be profiled against all reference sound waves, and using an upper threshold the remaining 'similarity' values would be averaged for each pariwise comparison, and the lowest value would indicate the best match.
</p>


<p style="line-height:2">

The matrix profile value would indicate, where global minima occur, that frequency pattern occurs **somewhere** in the reference species audio.
![]('./assets/discords-motifs.png')
</p>




<p style="line-height:2">

Using an upper threshold would ideally serve as a noise reducer, since a thunder rumble in the backround would cause a high profile distance for that period of time. However this could comprimise the overall level of information, so in the future ideally noise would be removed with traditional methods and then profiling would take place. Especially if 'synthetic-discords' exists, or audio which is not from a bird in this case and completly overides the information

![]('./assets/synthetic-discords.png')
</p>
<p style="line-height:2">

This method differs with tradtional methods with respect to dimensional complexity. The profile would be treating the raw audio as a 2 dimensional array, wheras a FFT transform would be modeling on a 3 dimensional array. So perhaps that makes up for the computational demand.
</p>



<p style="line-height:2">

Lastly, the advantage of this technique is that the field recording does not need to be indexed where the bird call occurs, and reference do not need to be the same time duration so long as the sequence length is not greater than either. In this example the bassline occurs at different time periods, but is identified.
![]('./assets/example1.png')

but you can also imagine a scenario where a longer song is compared to just the 10 second sample of the baseline. This would also work. Here's another example of comparing these two songs, in the stumpy documentation [link](https://stumpy.readthedocs.io/en/latest/Tutorial_AB_Joins.html)
</p>



# Problem

Identify bird species from raw audio file.

Your challenge in this competition is to identify which birds are calling in long recordings given quite limited training data. This is the exact challenge faced by scientists trying to monitor rare birds in Hawaii. For example, there are only a few thousand individual Nene geese left in the world, which makes it difficult to acquire recordings of their calls.

This competition uses a hidden test. When your submitted notebook is scored, the actual test data (including a sample submission) will be availabe to your notebook.

Files
- train_metadata.csv - A wide range of metadata is provided for the training data. The most directly relevant fields are:

- primary_label - a code for the bird species. You can review detailed information about the bird codes by appending the code to https://ebird.org/species/, such as https://ebird.org/species/amecro for the American Crow.
- secondary_labels: Background species as annotated by the recordist. An empty list does not mean that no background birds are audible.
- author - the eBird user who provided the recording.
- filename: the associated audio file.
- rating: Float value between 0.0 and 5.0 as an indicator of the quality rating on Xeno-canto and the number of background species, where 5.0 is the highest and 1.0 is the lowest. 0.0 means that this recording has no user rating yet.
- train_audio/ - The bulk of the training data consists of short recordings of individual bird calls generously uploaded by users of xenocanto.org. These files have been downsampled to 32 kHz where applicable to match the test set audio and converted to the ogg format.
- test_soundscapes/ - When you submit a notebook, the test_soundscapes directory will be populated with approximately 5,500 recordings to be used for scoring. These are each within a few milliseconds of 1 minute long and in the ogg audio format. Only one soundscape is available for download.

- test.csv - Metadata for the test set. Only the first three rows are available for download; the full test.csv is provided in the hidden test set.

- row_id - A unique identifier for the row.
- file_id - A unique identifier for the audio file.
- bird - The ebird code for the row. There is one row for each of the scored species per 5 second window per audio file.
- end_time - The last second of the 5 second time window (5, 10, 15, etc).
- sample_submission.csv - A valid sample submission. Only the first three rows are available for download; the full submission.csv is provided in the hidden test set.

- row_id - A unique identifier for the row.
- target - True/False for whether or not the bird in question called during the 5 second window.
- scored_birds.json - The subset of the species in the dataset that are scored.

- eBird_Taxonomy_v2021.csv - Data on the relationships between different species.
