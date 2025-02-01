import sys
import time
import pytz
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List

from utils import (
    get_daily_papers_by_keyword_with_retries,
    generate_table,
    back_up_files,
    restore_files,
    remove_backups,
    get_daily_date,
    write_papers_to_file,
    count_papers_by_keyword,
    write_keyword_statistics,
    generate_monthly_stats_plot,
    read_all_existing_papers,
)


HomeURL = "https://arxcompass.github.io"


def main(keywords: List[str], max_result: int, column_names: List[str]):
    try:
        beijing_timezone = pytz.timezone("Asia/Singapore")
        current_date = datetime.now(beijing_timezone)

        # Create papers directory if it doesn't exist
        os.makedirs("papers", exist_ok=True)
        for keyword in keywords:
            os.makedirs(
                os.path.join("papers", keyword.replace(" ", "_").lower()), exist_ok=True
            )

        # Backup files before making any changes
        back_up_files()

        # Read existing papers
        existing_papers = read_all_existing_papers()

        # request new papers
        for keyword in keywords:
            if len(keyword.split()) == 1:
                link = "AND"  # for keyword with only one word
            else:
                link = "OR"
            papers = get_daily_papers_by_keyword_with_retries(
                keyword, column_names, max_result, link
            )
            if papers is None:
                raise Exception(f"Failed to get papers for keyword: {keyword}")

            papers = sorted(papers, key=lambda x: x["Date"], reverse=True)

            existing_papers_by_keyword_links = [
                paper["Link"] for paper in existing_papers[keyword]
            ]
            # merge papers with existing papers by link
            new_papers = [
                paper
                for paper in papers
                if paper["Link"] not in existing_papers_by_keyword_links
            ]

            # write papers to files
            filepath = write_papers_to_file(
                HomeURL, new_papers, existing_papers[keyword], keyword, current_date
            )
            if filepath is None:
                raise Exception(f"Failed to write papers for keyword: {keyword}")

        # Write to README.md
        with open("README.md", "w") as f_rm:
            f_rm.write("# ArXCompass - Navigating the AI Research Frontier\n")
            f_rm.write(
                "An intelligent system that monitors and curates cutting-edge AI research papers from arXiv, guiding you toward the most impactful discoveries.\n\n"
            )
            f_rm.write(
                "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)\n\n"
            )
            f_rm.write("## Features\n\n")
            f_rm.write(
                "- 🔄 **Real-time Updates**: Fresh research papers delivered daily\n"
            )
            f_rm.write(
                "- 🎯 **Topic-focused**: Precisely curated for your research interests\n"
            )
            f_rm.write(
                "- 📊 **Research Analytics**: Track publication trends and patterns\n"
            )
            f_rm.write(
                "- 🗂️ **Smart Organization**: Papers neatly categorized by topics and dates\n"
            )
            f_rm.write(
                "- 📱 **Mobile-friendly**: Access your research feed anywhere\n\n"
            )

            f_rm.write("## Quick Links\n\n")
            for keyword in keywords:
                keyword_path = f"papers/{keyword.replace(' ', '_').lower()}"
                f_rm.write(f"- [{keyword}]({keyword_path}/)\n")
            f_rm.write("\n")

            f_rm.write("## How to Use\n\n")
            f_rm.write(
                "1. Click 'Watch' in the top right to receive daily notifications\n"
            )
            f_rm.write("2. Browse papers by topic in the Quick Links section\n")
            f_rm.write("3. View statistics and trends in each topic's README\n\n")

            f_rm.write(f"Last update: {current_date.strftime('%Y-%m-%d')}\n\n")

            # Add paper statistics
            f_rm.write("## Statistics\n\n")
            f_rm.write("| Research Topic | Total Papers | Latest Month |\n")
            f_rm.write("| --- | --- | --- |\n")

            for keyword in keywords:
                stats = count_papers_by_keyword(keyword)
                latest_month = next(iter(stats["months"].items()), (None, 0))
                latest_month_str = (
                    f"{latest_month[0]} ({latest_month[1]} papers)"
                    if latest_month[0]
                    else "No papers"
                )
                f_rm.write(f"| {keyword} | {stats['total']} | {latest_month_str} |\n")

            f_rm.write("\n## Monthly Trends\n\n")

            # Generate and save plots for each keyword
            for keyword in keywords:
                stats = count_papers_by_keyword(keyword)
                if stats["total"] > 0:
                    fig = generate_monthly_stats_plot(stats, keyword)
                    plot_path = os.path.join(
                        "papers", keyword.replace(" ", "_").lower(), "monthly_stats.png"
                    )
                    fig.savefig(plot_path)
                    plt.close(fig)
                else:
                    print(f"No papers found for keyword: {keyword}")

                # Write statistics markdown file
                stats_path = os.path.join(
                    "papers", keyword.replace(" ", "_").lower(), "README.md"
                )
                with open(stats_path, "w", encoding="utf-8") as f:
                    f.write(f"# Statistics for {keyword}\n\n")

                    # Add navigation breadcrumb
                    f.write(
                        f"[Home]({HomeURL}) - [Papers]({HomeURL}/papers) - [{keyword}]({HomeURL}/papers/{keyword.replace(' ', '_').lower()})\n\n"
                    )

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
                        percentage = (
                            (count / stats["total"] * 100) if stats["total"] > 0 else 0
                        )
                        f.write(
                            f"| [{month}](./{month}/papers_1.md) | {count} | {percentage:.1f}% |\n"
                        )

                    # Add plot to README
                    relative_path = os.path.relpath(plot_path)
                    f_rm.write(f"### {keyword}\n\n")
                    f_rm.write(
                        f"![Monthly Paper Counts for {keyword}]({relative_path})\n\n"
                    )

            # Write README.md to papers/README.md
            with open("papers/README.md", "w", encoding="utf-8") as f:
                f.write(f"# ArXCompass - Navigating the AI Research Frontier\n")
                f.write(
                    f"An intelligent system that monitors and curates cutting-edge AI research papers from arXiv, guiding you toward the most impactful discoveries.\n\n"
                )
                f.write(f"## Keywords\n\n")
                for keyword in keywords:
                    keyword_path = f"{keyword.replace(' ', '_').lower()}"
                    f.write(f"- [{keyword}]({keyword_path}/)\n")
                f.write("\n")
                f.write(f"Last update: {current_date.strftime('%Y-%m-%d')}\n\n")
                f.write(f"## Statistics\n\n")
                f.write(f"| Research Topic | Total Papers | Latest Month |\n")
                f.write(f"| --- | --- | --- |\n")
                for keyword in keywords:
                    stats = count_papers_by_keyword(keyword)
                    latest_month = next(iter(stats["months"].items()), (None, 0))
                    latest_month_str = (
                        f"{latest_month[0]} ({latest_month[1]} papers)"
                        if latest_month[0]
                        else "No papers"
                    )
                    f.write(f"| {keyword} | {stats['total']} | {latest_month_str} |\n")
                f.write("\n")
                f.write(f"## Monthly Trends\n\n")
                for keyword in keywords:
                    keyword_path = f"{keyword.replace(' ', '_').lower()}"
                    f.write(f"### {keyword}\n\n")
                    f.write(
                        f"![Monthly Paper Counts for {keyword}]({keyword_path}/monthly_stats.png)\n\n"
                    )

        # If everything succeeded, remove backups
        remove_backups()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        restore_files()
        sys.exit(1)


if __name__ == "__main__":
    keywords = ["gaussian splatting", "embodied ai", "llm"]
    max_result = 200
    column_names = ["Title", "Link", "Abstract", "Date", "Comment"]
    main(keywords, max_result, column_names)
