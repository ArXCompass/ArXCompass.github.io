import os
import time
import pytz
import shutil
import datetime
from typing import List, Dict
import urllib, urllib.request
import re

import feedparser
from easydict import EasyDict
import matplotlib.pyplot as plt


def remove_duplicated_spaces(text: str) -> str:
    return " ".join(text.split())


def request_paper_with_arXiv_api(
    keyword: str, max_results: int, link: str = "OR"
) -> List[Dict[str, str]]:
    # keyword = keyword.replace(" ", "+")
    assert link in ["OR", "AND"], "link should be 'OR' or 'AND'"
    keyword = '"' + keyword + '"'
    url = "http://export.arxiv.org/api/query?search_query=ti:{0}+{2}+abs:{0}&max_results={1}&sortBy=lastUpdatedDate".format(
        keyword, max_results, link
    )
    url = urllib.parse.quote(url, safe="%/:=&?~#+!$,;'@()*[]")
    response = urllib.request.urlopen(url).read().decode("utf-8")
    feed = feedparser.parse(response)

    # NOTE default columns: Title, Authors, Abstract, Link, Tags, Comment, Date
    papers = []
    for entry in feed.entries:
        entry = EasyDict(entry)
        paper = {}  # Changed from EasyDict() to regular dict

        # title
        paper["Title"] = remove_duplicated_spaces(entry.title.replace("\n", " "))
        # abstract
        paper["Abstract"] = remove_duplicated_spaces(entry.summary.replace("\n", " "))
        # authors
        paper["Authors"] = [
            remove_duplicated_spaces(_["name"].replace("\n", " "))
            for _ in entry.authors
        ]
        # link
        paper["Link"] = remove_duplicated_spaces(entry.link.replace("\n", " "))
        # tags
        paper["Tags"] = [
            remove_duplicated_spaces(_["term"].replace("\n", " ")) for _ in entry.tags
        ]
        # comment
        paper["Comment"] = remove_duplicated_spaces(
            entry.get("arxiv_comment", "").replace("\n", " ")
        )
        # date
        try:
            # ArXiv dates are in format like '2024-01-25T19:37:43Z'
            paper["Date"] = datetime.datetime.strptime(
                entry.updated, "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%Y-%m-%d")
        except:
            paper["Date"] = (
                entry.updated
            )  # fallback to original format if parsing fails

        papers.append(paper)
    return papers


def filter_tags(
    papers: List[Dict[str, str]], target_fileds: List[str] = ["cs", "stat"]
) -> List[Dict[str, str]]:
    # filtering tags: only keep the papers in target_fileds
    results = []
    for paper in papers:
        tags = paper["Tags"]  # Changed from paper.Tags
        for tag in tags:
            if tag.split(".")[0] in target_fileds:
                results.append(paper)
                break
    return results


def get_daily_papers_by_keyword_with_retries(
    keyword: str,
    column_names: List[str],
    max_result: int,
    link: str = "OR",
    retries: int = 6,
) -> List[Dict[str, str]]:
    for _ in range(retries):
        papers = get_daily_papers_by_keyword(keyword, column_names, max_result, link)
        if len(papers) > 0:
            return papers
        else:
            print("Unexpected empty list, retrying...")
            time.sleep(60 * 30)  # wait for 30 minutes
    # failed
    return None


def get_daily_papers_by_keyword(
    keyword: str, column_names: List[str], max_result: int, link: str = "OR"
) -> List[Dict[str, str]]:
    # get papers
    papers = request_paper_with_arXiv_api(
        keyword, max_result, link
    )  # NOTE default columns: Title, Authors, Abstract, Link, Tags, Comment, Date
    # NOTE filtering tags: only keep the papers in cs field
    # TODO filtering more
    papers = filter_tags(papers)
    # select columns for display
    papers = [
        {column_name: paper[column_name] for column_name in column_names}
        for paper in papers
    ]
    return papers


def generate_table(papers: List[Dict[str, str]]) -> str:
    """Generate markdown table with papers in OpenReview bid style"""
    if not papers:
        return "No papers found.\n"

    table = ''
    for paper in papers:
        table += f'<div class="paper-card">\n'
        table += f'    <div class="paper-title"><a href="{paper["Link"]}">{paper["Title"]}</a></div>\n'
        table += f'    <div class="paper-meta">\n'
        table += f'      📅 {paper["Date"]}\n'
        if paper.get("Comment"):
            table += f'      | 💬 {paper["Comment"]}\n'
        table += f'    </div>\n'
        table += f'    <details class="paper-abstract">\n'
        table += f'      {paper["Abstract"]}\n'
        table += f'    </details>\n'
        table += f'</div>\n'

    return table


def back_up_files():
    """Back up README.md and create if it doesn't exist"""
    if os.path.exists("README.md"):
        shutil.copy2("README.md", "README.md.bk")
    else:
        # Create empty README if it doesn't exist
        with open("README.md", "w") as f:
            f.write("# Daily arXiv Papers\n")
        shutil.copy2("README.md", "README.md.bk")


def restore_files():
    """Restore README.md from backup if it exists"""
    if os.path.exists("README.md.bk"):
        try:
            if os.path.exists("README.md"):
                os.remove("README.md")
            shutil.move("README.md.bk", "README.md")
            print("Restored files from backup")
        except Exception as e:
            print(f"Error restoring backup: {str(e)}")
            if os.path.exists("README.md.bk"):
                print("Backup file still exists at README.md.bk")


def remove_backups():
    """Safely remove backup files"""
    try:
        if os.path.exists("README.md.bk"):
            os.remove("README.md.bk")
    except Exception as e:
        print(f"Warning: Could not remove backup file: {str(e)}")


def get_daily_date():
    # get beijing time in the format of "March 1, 2021"
    beijing_timezone = pytz.timezone("Asia/Singapore")
    today = datetime.datetime.now(beijing_timezone)
    return today.strftime("%B %d, %Y")


def get_paper_directory(keyword: str, paper_date: str) -> str:
    """Create and return path to directory for storing papers by keyword and upload date"""
    # Convert keyword to valid directory name
    keyword_dir = keyword.replace(" ", "_").lower()

    # Parse paper date (format: YYYY-MM-DD)
    date = datetime.datetime.strptime(paper_date, "%Y-%m-%d")

    # Create path like papers/gaussian_splatting/2025_01/
    paper_dir = os.path.join("papers", keyword_dir, f"{date.year}_{date.month:02d}")

    # Create directories if they don't exist
    os.makedirs(paper_dir, exist_ok=True)

    return paper_dir


def write_papers_to_file(
    home_url: str,
    new_papers: List[Dict[str, str]],
    existing_papers: List[Dict[str, str]],
    keyword: str,
    _: datetime,
) -> str:
    """Write papers to markdown files organized by keyword and upload date, with max 100 papers per file"""
    # Sort papers by date in descending order (newest first)
    sorted_papers = sorted(new_papers, key=lambda x: x["Date"], reverse=True)

    # Group papers by month
    paper_groups = {}
    for paper in sorted_papers:
        paper_date = paper["Date"]
        paper_dir = get_paper_directory(keyword, paper_date)

        # get existing papers in the same month
        existing_papers_in_month = [
            paper
            for paper in existing_papers
            if f"{paper['Date'].split('-')[0]}_{paper['Date'].split('-')[1]}"
            == paper_dir.split("/")[-1]
        ]
        if paper_dir not in paper_groups:
            paper_groups[paper_dir] = existing_papers_in_month
        paper_groups[paper_dir].append(paper)

    # If no new papers were added, return the most recent existing file
    if not paper_groups:
        # Find the most recent file
        keyword_dir = keyword.replace(" ", "_").lower()
        base_dir = os.path.join("papers", keyword_dir)
        if not os.path.exists(base_dir):
            # If no papers exist at all, create an empty file
            paper_dir = get_paper_directory(keyword, sorted_papers[0]["Date"])
            filepath = os.path.join(paper_dir, "papers_1.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {keyword}\n\n")
                table = generate_table(sorted_papers)
                f.write(table)
            return filepath

        # Find most recent month directory
        months = [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("20")
        ]
        months.sort(reverse=True)

        if not months:
            return None

        recent_dir = os.path.join(base_dir, months[0])
        files = [
            f
            for f in os.listdir(recent_dir)
            if f.endswith(".md") and f.startswith("papers_")
        ]
        if not files:
            return None

        # Return first file (papers_1.md)
        return os.path.join(recent_dir, "papers_1.md")

    # Write each month's papers to files
    first_filepath = None
    for paper_dir, month_papers in paper_groups.items():
        # Sort all papers by date
        month_papers = sorted(month_papers, key=lambda x: x["Date"], reverse=True)

        # Split papers into chunks of 100
        papers_per_file = 100
        num_files = (len(month_papers) + papers_per_file - 1) // papers_per_file

        for i in range(num_files):
            start_idx = i * papers_per_file
            end_idx = min((i + 1) * papers_per_file, len(month_papers))
            current_papers = month_papers[start_idx:end_idx]

            # Generate filename (always use numbered format)
            filename = f"papers_{i+1}.md"
            filepath = os.path.join(paper_dir, filename)

            # Store first filepath to return (for README link)
            if first_filepath is None:
                first_filepath = filepath

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {keyword} - {paper_dir.split('/')[-1]}\n\n")
                f.write("## Navigation\n\n")
                f.write(
                    f"[Home]({home_url}) / [Papers]({home_url}/papers) / [{keyword}]({home_url}/papers/{keyword.replace(' ', '_').lower()})\n\n"
                )
                for j in range(num_files):
                    nav_filename = f"papers_{j+1}.md"
                    if j == i:
                        f.write(f"- Part {j+1}\n")
                    else:
                        f.write(f"- [Part {j+1}]({nav_filename})\n")
                f.write("\n## Papers\n\n")

                table = generate_table(current_papers)
                f.write(table)

    return first_filepath


def read_existing_papers(paper_dir: str) -> List[Dict[str, str]]:
    """Read existing papers from all markdown files in the directory"""
    existing_papers = []
    if not os.path.exists(paper_dir):
        return existing_papers

    # Read all markdown files in directory
    for filename in os.listdir(paper_dir):
        if not filename.endswith(".md"):
            continue

        filepath = os.path.join(paper_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract paper information from paper cards
        paper_cards = content.split('<div class="paper-card">')
        for card in paper_cards[1:]:  # Skip first split which is before first card
            paper = {}
            
            # Extract title and link
            title_match = re.search(r'<a href="([^"]+)">([^<]+)</a>', card)
            if title_match:
                paper["Link"] = title_match.group(1)
                paper["Title"] = title_match.group(2)
            
            # Extract date and comment
            meta_section = re.search(r'<div class="paper-meta">\s*(.*?)\s*</div>', card, re.DOTALL)
            if meta_section:
                meta_text = meta_section.group(1)
                
                # Extract date
                date_match = re.search(r'📅\s*(\d{4}-\d{2}-\d{2})', meta_text)
                if date_match:
                    paper["Date"] = date_match.group(1)
                
                # Extract comment if present
                comment_match = re.search(r'💬\s*(.*?)(?:\n|$)', meta_text)
                if comment_match:
                    paper["Comment"] = comment_match.group(1).strip()
                else:
                    paper["Comment"] = ""
            
            # Extract abstract
            abstract_match = re.search(r'<details class="paper-abstract">\s*(.*?)\s*</details>', card, re.DOTALL)
            if abstract_match:
                paper["Abstract"] = abstract_match.group(1).strip()
            
            # Only add paper if we have at least title and link
            if "Title" in paper and "Link" in paper:
                existing_papers.append(paper)

    return existing_papers


def filter_new_papers(
    new_papers: List[Dict[str, str]], existing_papers: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Filter out papers that already exist in the directory"""
    # Create set of existing links, skip papers without links
    existing_links = {
        paper.get("Link", "") for paper in existing_papers if "Link" in paper
    }
    # Only keep papers that have a link and aren't in existing_links
    return [
        paper
        for paper in new_papers
        if paper.get("Link") and paper["Link"] not in existing_links
    ]


def count_papers_by_keyword(keyword: str) -> Dict[str, int]:
    """Count total papers and papers by month for a given keyword"""
    keyword_dir = keyword.replace(" ", "_").lower()
    base_dir = os.path.join("papers", keyword_dir)

    if not os.path.exists(base_dir):
        return {"total": 0, "months": {}}

    stats = {"total": 0, "months": {}}

    # Get all month directories
    months = sorted(os.listdir(base_dir), reverse=True)

    for month in months:
        month_dir = os.path.join(base_dir, month)
        month_papers = []

        # if month_dir is not a directory, skip
        if not os.path.isdir(month_dir):
            continue

        # Read all markdown files in the month directory
        for filename in os.listdir(month_dir):
            if not filename.endswith(".md"):
                continue

            # Read papers from this file
            month_papers.extend(read_existing_papers(month_dir))

        # Add month stats
        if month_papers:
            stats["months"][month] = len(month_papers)
            stats["total"] += len(month_papers)

    return stats


def generate_monthly_stats_plot(stats: Dict[str, int], keyword: str):
    """Generate a bar plot of monthly paper counts"""
    # Sort months in chronological order
    months = sorted(stats["months"].keys())
    counts = [stats["months"][month] for month in months]

    # Create the plot
    fig = plt.figure(figsize=(12, 6))
    plt.bar(months, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title(f'Monthly Paper Counts for "{keyword}"')
    plt.xlabel("Month")
    plt.ylabel("Number of Papers")

    # Add value labels on top of each bar
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha="center", va="bottom")

    plt.tight_layout()
    return fig


def write_keyword_statistics(keyword: str, stats: Dict[str, int]):
    """Write detailed statistics for a keyword to a markdown file"""
    keyword_dir = keyword.replace(" ", "_").lower()
    base_dir = os.path.join("papers", keyword_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Generate and save monthly stats plot
    fig = generate_monthly_stats_plot(stats, keyword)
    plot_path = os.path.join(base_dir, "monthly_stats.png")
    fig.savefig(plot_path)
    plt.close(fig)

    # Write statistics markdown file
    stats_path = os.path.join(base_dir, "README.md")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"# Statistics for {keyword}\n\n")

        # Overall statistics
        f.write("## Overall Statistics\n\n")
        f.write(f"- Total number of papers: {stats['total']}\n")
        f.write(f"- Number of months tracked: {len(stats['months'])}\n")
        if stats["total"] > 0:
            avg_papers = stats["total"] / len(stats["months"])
            f.write(f"- Average papers per month: {avg_papers:.1f}\n")

        # Monthly trend visualization
        f.write("\n## Monthly Trends\n\n")
        f.write(f"![Monthly Paper Counts](monthly_stats.png)\n\n")

        # Detailed monthly breakdown
        f.write("## Monthly Breakdown\n\n")
        f.write("| Month | Paper Count | Percentage of Total |\n")
        f.write("| --- | --- | --- |\n")

        # Sort months in reverse chronological order
        sorted_months = sorted(stats["months"].keys(), reverse=True)
        for month in sorted_months:
            count = stats["months"][month]
            percentage = (count / stats["total"] * 100) if stats["total"] > 0 else 0
            f.write(f"| {month} | {count} | {percentage:.1f}% |\n")


def read_all_existing_papers() -> Dict[str, List[Dict[str, str]]]:
    """Read all existing papers for all keywords"""
    papers_by_keyword = {}

    if not os.path.exists("papers"):
        return papers_by_keyword

    # Iterate through keyword directories
    for keyword_dir in os.listdir("papers"):
        base_dir = os.path.join("papers", keyword_dir)
        if not os.path.isdir(base_dir):
            continue

        all_papers = []
        # Read papers from each month directory
        for month_dir in os.listdir(base_dir):
            month_path = os.path.join(base_dir, month_dir)
            if os.path.isdir(month_path):
                all_papers.extend(read_existing_papers(month_path))

        papers_by_keyword[keyword_dir.replace("_", " ")] = all_papers

    return papers_by_keyword
