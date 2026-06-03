---
layout: default
title: Home
---
# Hi, I'm Chiril Calin!

This site is dedicated to any interesting publications or blog posts I do! 

## Featured Projects / Publications

### [Untangling Tasks in a Toy Transformer](https://github.com/chirilcalin/untangling_tasks)
This project explores training a model for two simple tasks: summation, and concatenation, and investigates how we can intervene in our model's thinking to selectively disable one task, but not the other. 

### [Using sentiment analysis to quantify the relative desirability and acceptability of drug-product attributes](https://academic.oup.com/jamiaopen/article/9/3/ooag075/8699116)
Sentiment analysis performed on Pfizer customer complaint data around oral drug attributes, such as taste, shape, swallowability, etc. to guide industry direction for patient preference in drug formulation. Published June 1st 2026 in JAMIA Open Vol 9 Issue 3. Code is available on my [Pfizer Opensource repo](https://github.com/pfizer-opensource/tio2-sentiment-analysis).

## Recent Writing
<ul class="post-list">
  {% for post in site.posts limit:3 %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <span class="post-meta">{{ post.date | date: "%B %d, %Y" }}</span>
    </li>
  {% endfor %}
</ul>
