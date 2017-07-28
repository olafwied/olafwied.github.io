---
layout: page
title: Welcome to YARDS
tagline: Yet Another Resource on Data Science
supporting tagline: Yet Another Resource on Data Science
---
{% include JB/setup %}

My personal collection of concepts, ideas and practical tips about Data Science.

## Posts

Here you will find the most recent posts.

<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>

### About me
If you would like to get in touch with me, please find me on [linkedIn](https://www.linkedin.com/in/olafwied).

#

Check out [my profile on Data Science Stackexchange](https://datascience.stackexchange.com/users/23305/ow).

![Profile for YARDS author at Data Science SE, Q&A for Data science professionals and Machine Learning specialists](https://datascience.stackexchange.com/users//flair/23305.png "Profile for YARDS author at 'Data Science SE'")
