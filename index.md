---
layout: page
title: Welcome to YARDS
tagline: Yet Another Resource on Data Science
---
{% include JB/setup %}

This is my personal blog where I share my experiences implementing machine learning algorithms from scratch (for mostly educational purposes).

## Posts

Here you will find the most recent posts.

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>

### About me
If you would like to get in touch with me, please find me on [linkedIn](https://www.linkedin.com/in/olafwied).