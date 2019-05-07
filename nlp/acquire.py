import os
import json

import requests
from bs4 import BeautifulSoup


# BLOG ARTICLES
BASEURL = "https://codeup.com/blog"
HEADERS = {"User-Agent": "Codeup Ada Data Science"}
CACHE_FILENAME = "blog_posts.json"


def article_info(url):
    soup = page_soup(url)

    title = soup.find("h1").text
    author = soup.find("a", rel="author").text
    date_posted = soup.find("time", class_="mk-post-date")
    content = soup.find("div", class_="mk-single-content").text
    for span in soup.find_all(
        "span"
    ):  # sometimes the author is in the article text
        if "By " in span.text:
            author = span.text[3:]

    return dict(
        title=title,
        author=author,
        date=date_posted["datetime"],
        original=content,
    )


def all_page_links(soup):
    title_tags = soup.find_all("h3", class_="the-title")
    for tag in title_tags:
        yield tag.a["href"]


def all_page_articles(soup):
    return [article_info(link) for link in all_page_links(soup)]


def page_soup(url):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup


def get_last_page(soup):
    return int(soup.find(class_="pagination-max-pages").text.strip())


def get_blog_articles(use_cache=True):

    if use_cache and os.path.exists(CACHE_FILENAME):
        with open(CACHE_FILENAME, "r") as f:
            articles = json.load(f)

        return articles
    else:
        url = f"{BASEURL}"
        soup = page_soup(url)
        last_page = get_last_page(soup)
        articles = []
        for page_num in range(1, last_page + 1):
            url = f"{BASEURL}/page/{page_num}"
            soup = page_soup(url)

            articles += all_page_articles(soup)

        with open(CACHE_FILENAME, "w") as f:
            json.dump(articles, f)

        return articles


# INSHORTS
URL_PREFIX = "https://inshorts.com/en/read"


def page_contents(topic, use_cache):
    page = topic + ".txt"
    if use_cache and os.path.exists(page):
        with open(page, "r") as f:
            contents = f.read()
        return contents
    else:
        response = requests.get(url)
        with open(page, "w") as f:
            f.write(response.text)
        return response.text


def news_card(card, topic):
    title = card.find("span", itemprop="headline").text
    content = card.find("div", itemprop="articleBody").text
    category = topic

    return dict(title=title, original=content, category=category)


def get_cards(topic, use_cache):
    soup = BeautifulSoup(page_contents(topic, use_cache), "html.parser")
    return soup.find_all("div", class_="news-card")


def get_news_articles(use_cache=True):
    topics = ("business", "sports", "technology", "entertainment")
    url = "https://inshorts.com/en/read/"

    for t in topics:
        for card in get_cards(t, use_cache):
            yield news_card(card, t)
