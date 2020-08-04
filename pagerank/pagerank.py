import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )
    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # If page has at least one outgoing link
    if corpus[page]:
        # Initialise probability distribution to P(page chosen at random out of all pages in corpus)
        total_probabilities = [(1 - damping_factor) / len(corpus)] * len(corpus)
        total_probabilities_dict = dict(zip(corpus.keys(), total_probabilities))

        # Add additional probability for all pages linked to by current page
        link_probabilities = damping_factor / len(corpus[page])
        for link in corpus[page]:
            total_probabilities_dict[link] += link_probabilities
        return total_probabilities_dict

    # If page has no outgoing links, probability distribution chooses randomly among all pages with equal probability
    else:
        return dict(zip(corpus.keys(), [1 / len(corpus)] * len(corpus)))


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialise pagerank dictionary and set all values to 0
    pageranks = dict(zip(corpus.keys(), [0] * len(corpus)))

    # Start with random page
    page = random.choice(list(corpus.keys()))

    # Sample repeatedly for n total times (including initial random sample)
    # For each sample, increment count for current page and choose next page based on transition model
    for _ in (range(n - 1)):
        pageranks[page] += 1
        prob_distribution = transition_model(corpus, page, damping_factor)
        page = random.choices(list(prob_distribution.keys()), prob_distribution.values())[0]

    # Divide all page counts by n to get proportion of samples for that page
    pageranks = {page: num_samples/n for page, num_samples in pageranks.items()}

    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_pages = len(corpus)
    pageranks = dict(zip(corpus.keys(), [1/total_pages] * total_pages))
    pagerank_changes = dict(zip(corpus.keys(), [math.inf] * total_pages))

    # Keep updating pageranks until no pagerank value changes by > 0.001 between iterations
    while any(pagerank_change > 0.001 for pagerank_change in pagerank_changes.values()):
        for page in pageranks.keys():
            link_probability = 0
            for link_page, links in corpus.items():
                # Page with no links interpreted as having one link for every page in corpus
                if not links:
                    links = corpus.keys()
                if page in links:
                    link_probability += pageranks[link_page] / len(links)
            new_pagerank = ((1 - damping_factor) / total_pages) + (damping_factor * link_probability)

            # Keep track of change bewteen old and new pageranks, and store new pagerank
            pagerank_changes[page] = abs(new_pagerank - pageranks[page])
            pageranks[page] = new_pagerank
    
    return pageranks


if __name__ == "__main__":
    main()
