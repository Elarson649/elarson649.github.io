---
layout: post
title: Am I A One-Hit Wonder?
---

For my last project, I explored the science behind one-hit wonders, or artists who only are only known for one song- think of songs like The Macarena, Come On Eileen, or, more recently, The Harlem Shake! Since the beginning of Metis, I knew I wanted to do some sort of project with music and use the [Spotify API](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/), which provided numerical representations of the audio qualities of any song on Spotify (e.g. how danceable it is, the tempo, its loudness). I was particularly interested in one-hit wonders because as a pop music enthusiast, the phenomenom seemed pretty random! With the help of natural language processing and classification techniques, I wanted to see whether I could make some sense of one-hit wonders and figure out their ingredients.

insert photo of the ketchup song dance with link to https://www.youtube.com/watch?v=AMT698ArSfQ

For this project, I defined a one-hit wonder as an artist who only had one song reach the Billboard top 40. This definition comes from one-hit wonder expert and music journalist Wayne Jancik, the author of The Billboard Book of One-Hit Wonders, and seems to be the most common definition used in music literature. With this definition and [a lovely dataset](https://data.world/kcmillersean/billboard-hot-100-1958-2017) of the Billboard weekly hot 100 from Sean Miller/data.world, I was off to the races! 

The first challenge was standardizing the performer column, since some songs had multiple artists, often in different capacities; for example, a song could be a duet, where the artists are featured equally, or have a featured artist, where there is a primary artist and a featured artist. The syntax used to designate this was all over the place and changed with time (e.g. saying 'duet' isn't cool anymore, so collaborations are now designated with an 'x', as in Shawn Mendes x Camila Cabello). In addition, some designators like 'and' and '&' could indicate a collaboration or band name! For simplicity's sake, if the performer had the word 'featuring' or 'with' in it, I designated the artist before the word 'featuring' as the primary artist and then indicated that the song had a featured artist. In every other scenario, I was more conservative and assumed the songs did not have a featured artist and only considered the first artist as the primary artist.

insert image of estelle and kanye

After that slight drama, I did a bunch of data manipulation to find songs that were eligible to be a one-hit wonder, or, in other words, an artist's first song to reach the Billboard top 40. I then added in data via the Spotify API, including the genres of the artist (e.g. hip hop, pop) and the aforementioned audio features. I also added in the song's lyrics, producers, and writers via the Genius API. I was particularly curious about the impact of working with acclaimed producers and writers (think of [Max Martin](https://en.wikipedia.org/wiki/Max_Martin), superproducer/writer to pop's biggest stars). As a result, I leveraged the Genius data to engineer some features indicating if the song was produced or written by a star, or someone who had appeared on at least 3 other potential one-hit wonders at the time the song was released. After adding in the APIs and performing some data validation, we had 1500 songs in our dataset, 48% of which were one-hit wonders, which I thought of as surprisingly high! Alas, it is better to have one hit wonder-ed than to never have one hit wonder-ed at all, right?

I really wanted to leverage some of the NLP (natural language processing) techniques I picked up at Metis in my analysis, so I performed some feature engineering around the lyrics of songs. I used VADER to perform sentiment analysis, leveraged spaCy for named entity recognition (e.g. is the song using more specific language as opposed to more vague languasge or platitudes), measured the reading difficulty of the lyrics using the Flesch Reading Ease, and quantified the variety of words in the lyrics through a custom metric.

insert some image- there's too much text

I also wanted to compare the potential one-hit wonders to the music landscape at the time of the song's release. To avoid data leakage, I compared each one hit wonder to the previous year's top 100 songs via the Billboard year-end charts. These charts are based off a song's position and longevity on the weekly charts from the entire year. I used a [dataset](https://github.com/walkerkq/musiclyrics), scraped by Kaylin Pavlik, that already had the lyrics of the top songs. After adding in audio features and genres via Spotify's API, I developed three metrics to measure how similar a song was to other music popular at the time. First, I calculcated the genre score of the song/artist, which is defined as a count of how many songs' artists from the year-end chart from the prior year had the same genre as the one-hit wonder's artist (e.g. the potential one hit wonder was country and 15 songs on the prior year's year-end charts were country, so the genre score is 15 points). I only looked at the genre with the highest score if the song's artist had multiple genres. I then calculated the lyric distance, or how the lyrics of the song compare to the prior year's top 100. I got a bit creative with this after clustering/DBScan wasn't returning great results; I performed PCA on the TFIDF vectorizer of the songs from the prior year, then calculate the euclidean distance of the one-hit wonder's from the center/average. The higher the lyric distance, the more unique a song's lyrics are relative to other popular songs at the time. Lastly, I performed a similar PCA and distance technique with the audio features, to see if the production of the song was very different from the current music landspace.

image image image

Now, it's time for the most satisfying part of the data science process- modeling! Throughout this process, I tried a bunch of classification models: logistic regression with lasso and/or ridge regularization, linear SVM, random forests, KNN, XGBoost, and a neural network with 2 hidden layers. I also tested out some interaction terms via polynomial transformations with logistic regression and, from the results, added a new feature to the dataset: the overall chart performance, which is the song's number of weeks spent on the chart divided by its peak position. This metric performed better than the number of weeks by itself and wasn't strongly correlated with any other feature.

Optimizing the F-score, the neural network worked best. All of the linear models also performed strongly. I then dove into finding out the ingredients of a one-hit wonder, or, in data science term, feature importance. For this, I used the logistic regression that performed well.

Below are the most important and statistically significant features that increase the song's probability of being a one-hit wonder:

picture of a food related one hit wonder, like laffy taffy or milkshake

If an artist is featured on the song! As an example, think of American Boy by Estelle ft. Kanye West or Latch by Disclosure ft. Sam Smith. I think this is pretty intuitive- if there's a featured artist, it's more difficult to say who drove the song's success.

Relatively poor performance on the Billboard weekly charts. This finding is also quite intuitive; if a song's peak position is closer to 40 than 1 and the song wasn't on the weekly charts for a long period of time, it's more likely going to be a one-hit wonder. Let's look at How You Remind Me by Nickelback! The song was on the weekly charts for 49 weeks and peaked at number one, decreasing the Nickelback's chance of being a one-hit wonder. On the other hand, let's look at Irish girl group B\*Witched and their surprisingly racy first single C'est L Vie (I hadn't heard the song since 1999- the things I can pick up on now!). The song peaked at number 9, but was only on the chart for 15 weeks- for a top ten song, this is a pretty short stay on the charts, so the song was more likely to be a one-hit wonder.

Image

If the lyrics are similar to other popular songs at the time, the song is more likely going to be a one-hit wonder. This was a surprising finding to me- when I think of one-hit wonders, I think of songs with odd lyrics (e.g. The Macarena, I'm Too Sexy, Barbie Girl, etc). These songs may be the exception, but in general, the more unorthodox the lyrics, the lower chance a song is going to be a one-hit wonder. An example of a song that had a low lyric distance score (aka it was similar to other songs released at the time) is [What Is Love by Haddaway](https://genius.com/Haddaway-what-is-love-lyrics). If you look at the lyrics, you can see they're a bit generic (but undeniably catchy).

Gif 

If the genre of the artist isn't popular at the song's time of release, the song is more likely going to be a one-hit wonder. An example of this is Ho Hey by The Lumineers. Released in 2012, The Lumineers are a folk/rock group. Only two songs' artists in 2011's year-end top 100 songs had these genres, increasing the song's probability of being a one-hit wonder!

If the song was released in the 2010s, the song is more likely going to be a one-hit wonder. My initial thought on this had to do with the shift in how we consume music over the past 20 years, from radio and CDs, to digital purchases, and, finally, to streaming. In addition, Billboard has been tweaking how they count streams in their rankings throughout the 2010s! My hypothesis is that all these changes have contributed to one-hit wonders being more likely in the 2010s, but I have to do a bit more research!

Lastly, some genres were more likely to be one-hit wonders than others. Artists categorized as R&B a relatively higher chance of being a one-hit wonder, while Country and Hip Hop artists were the least likely to be one-hit wonders.

I hope you enjoyed my analysis, and the next time you ponder why that artist you loved from a few years ago hasn't had much success lately, consider these findings! Finally, I'll leave you with my favorite one-hit wonder, one of the best dance songs of all time...


