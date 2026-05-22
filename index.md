---
layout: default
title: Home
---
# Hi, I'm Chiril Calin!

This site is dedicated to any interesting publications or blog posts I do! 

## Featured Projects

### [Untangling Tasks in a Toy Transformer](https://github.com/chirilcalin/untangling_tasks)
This project explores training a model for two simple tasks: summation, and concatenation, and investigates how we can intervene in our model's thinking to selectively disable one task, but not the other. 

### [Publications will be added soon.]

## Recent Writing
<ul class="post-list">
  {% for post in site.posts limit:3 %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <span class="post-meta">{{ post.date | date: "%B %d, %Y" }}</span>
    </li>
  {% endfor %}
</ul>
