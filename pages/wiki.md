---
layout: page
title: Experience
description: 人越学越觉得自己无知
keywords: 个人, experience
comments: false
menu: 个人
permalink: /wiki/
---

> 成长 是一种经历, 成熟 是一种阅历

<ul class="listing">
{% for wiki in site.wiki %}
{% if wiki.title != "Wiki Template" %}
<li class="listing-item"><a href="{{ wiki.url }}">{{ wiki.title }}</a></li>
{% endif %}
{% endfor %}
</ul>
