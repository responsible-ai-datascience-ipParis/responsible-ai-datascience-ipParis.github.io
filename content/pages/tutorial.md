---
title: Tutorial
---

How to create and publish your blogpost ?

<hr/>

The blogpost builds upon [Hugo](https://gohugo.io/) and [Markdown](https://www.markdownguide.org/).

For markdown you can easily find a cheat sheet [here](https://www.markdownguide.org/cheat-sheet/).  

0. Setup: you need to have a GitHub Account, a terminal and a text editor of your choice (e.g.VSCode, nano, gedit)

1. Install Hugo on your laptop. Hugo is available for all operating systems. You can find the installation guide [here](https://gohugo.io/installation/). Be careful, you need at least Hugo version 0.120 otherwise, this will not work!
To verify your version
```
hugo version
```

For Ubuntu, the default version is not the correct one, please download it directly from their
[github](https://github.com/gohugoio/hugo/releases/tag/v0.121.2).
- Download the `hugo_extended_0.121.2_linux-amd64.deb`.
- Using the terminal, go to the folder where the file is
- Then `sudo apt install ./hugo_extended_0.121.2_linux-amd64.deb`


2. Go to the following [repository](https://github.com/responsible-ai-datascience-ipParis/responsible-ai-datascience-ipParis.github.io) and `fork` it.

3. Open a terminal

4. Clone the forked repository of the course using the following command

```
git clone --recurse-submodules https://....
```

Replace the `https://` with your forked repository url.

<!---*Note*. **This is not a github tutorial,** but for each article please work on a separate branch to avoid breaking the whole thing. If you are not familiar with github good practices, discuss with your teammates !--->

5. With the terminal, navigate in the cloned repository

```
cd responsible-ai-datascience-ipParis.github.io
```

<!---
5. Create a branch named adding-my-post. **Replace my-post** with some keywords related to the paper that you are working on.

```
git checkout -b adding-my-post
```

*Note*: the `-b` option is there because we assume that this branch did not exist.
-->
6. Run the hugo server command and visit [http://localhost:1313](http://localhost:1313) in your browser.
```
hugo server
```

7. Let’s create a new blog post by running the following command. Once again, please replace `choose-a-name` by a name close to your article.

```
hugo new posts/choose-a-name.md
```

8. This will create a markdown file named `my-first-blog.md` in the `blog` folder underneath the `content` folder.

If you open this file, you will see that it is not empty. Hugo automatically created some metadata. Hugo has prefilled the title field from the name of the Markdown file, the timestamp of when the file was created, and it has automatically put the file into draft mode.

9. Change the draft status to false (by default it is set to true) in the newly created file.

10. Write your blogpost by modifying the markdown file create during **step 7**.
This step should take you long enough ! Once again, to visualise the result of your progress, simply run the hugo server command and visit [http://localhost:1313](http://localhost:1313) in your browser.
For illustrations or image, create your own folder to avoid filenames confusion with other groups.

**DO NOT** touch anything else !

*Note*: regularly commit to your repository to save backups.
<!---*Note*: If you don't feel comfortable with git, we recommend not to push the work on the main branch until you are done here to avoid possible conflicts.--->

11. Once you are satisfied with your blogpost you can do the final push of your work !

```
git add .
git commit -m "i'm done'"
git push origin
```

12. Time to merge your forked repository with the original one !
- First, you need to update your fork by clinking on `Sync fork` button from github.
- Then, click on `Contribute` then `open pull request`!
<!---
Go to github website

   -  On the GitHub interface, click “Create & pull request”
   -  Give your PR an informative title and summary, then click “Create pull request”
   -  After the continuous integration test has passed, click “Merge pull request” then “Confirm merge”
--->
13. While waiting for CD/GitHub Pages to deploy the update, let’s update our local repo.

```
git checkout main
git pull
```
After a few minutes the website is updated and you can check your blogpost in the blog page of the website.
There is no need on your side to update the index page to list your blog post, this will be done by the supervisors of the course at the very end.

*Remark: This tutorial is a small part of the tutorial used to set up the blogpost website [link](https://carpentries-incubator.github.io/blogging-with-hugo-and-github-pages/05-add-blog-content/index.html) for which a detailed youtube video is available.*
