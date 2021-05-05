# DSC-Capstone
UMass Dartmouth Data Science Capstone Project - Repository for DSC 498/499 files

**Title:** Optical Character Recognition and Correction for Images of Digital Displays

**Abstract:** Data structures such as tables and lists are commonly presented through graphical user interfaces (GUIs). In many cases, accessing the data directly is inconvenient or prohibited by the platform itself (Apple iOS, Nintendo consoles, etc.) or unavailable locally as with video streaming. Notably, the data is not private to the user, but collection must be performed by hand. This project introduces a program for Optical Character Recognition (OCR) using Tesseract 4 and Python to convert data displayed in images or videos into discrete structures. Image transformations such as binarization, scaling, skewing, edge detection, and denoising enable compatibility with elaborate GUIs and photographs of digital displays. Predictive methods using Levenshtein and character-shape distances provide text correction to significantly improve output accuracy. Error analysis is also conducted for individual correction methods using several sample images, dictionaries, and fonts.

**Contact:** [Brian Cornet](mailto:bcornet@umassd.edu?subject=[GitHub]%20I%20clicked%20a%20thing)

**Instructor:** [Scott Field](mailto:sfield@umassd.edu?subject=[GitHub]%20DSC499%20Capstone%20%2D%20Brian%20Cornet)

## Structure
Relevant project files will be kept in the root directory. This includes the primary script "tess.py" and various sample images used in experimentation. Note that you'll need to install Tesseract OCR and PyTesseract in order to run this. Post-transformation images and graphs have been moved to the Output folder.

Check the Notes folder for weekly notes and doodles. Presentations contains various Microsoft PowerPoint Presentation (.pptx) and Microsoft Word Document (.docx) files both in their original formats and as .pdf files. As of May 5th, the latter folder takes up about 112 MB, so heads up. Also there's an Other folder which just contains random stuff from other classes because I'm too lazy to set up another git. 

## Presentation
[May 3rd, 12:51 PM as part of Data Science Capstone Day via Zoom meeting.](https://bitbucket.org/cscvr/data-science-capstone-day-umassd/wiki/Home)

Recording coming soon!

## What's the Deal?
I came up with a bunch of project ideas for the capstone, most of which I was hoping would be useful or enlightening to develop. This one was my favorite, so I went with it. Pretty sure my plan was to use this to make it easier to nerd out over my stats and inventories in video games. Bonus: being able to use images from stuff I like as examples! It's been a hoot thus far. The last semester was largely a matter of familiarizing myself with Tesseract OCR with some basic experiments involving image clean-up. Taking ECE 621 at the same time was super convenient too.

## Project To-Doots
- [x] Install Tesseract OCR and PyTesseract
- [x] Experiment with basic PyTesseract commands in Python
- [x] Investigate methods for improving OCR accuracy
- [x] Implement word and character highlighter
- [x] Implement bitplane seperator
- [x] Implement discrete wavelet transform method
- [x] Implement Levenshtein ratio/distance method
- [x] Prepare final presentation and report (DSC 498)
- [x] Implement xy-plane rotation method
- [x] Implement xz- and yz-plane skew methods
- [x] Implement noise removal method
- [x] Implement dilation/erosion methods
- [x] Implement border addition/removal methods
- [x] Experiment with automating correction methods
- [x] Experiment with off-screen text-based images
- [x] Experiment with user-defined dictionary support
- [x] Develop of image comparison methods
- [x] Prepare final presentation and report (DSC 499)

## Expected DSC499 Timeline Probably
- February
  1. xy-plane Rotation method
  2. xz- and yz- plane Skew methods
  3. Noise Removal method
  4. Dilation/Erosion methods
  5. Border Addition/Removal methods
- March
  1. Automated correction detection
  2. Off-screen text experimentation
  3. \(Optional) English and custom dictionary support
  4. \(Optional) Image comparison
- April
  1. Initial draft of presentation
  2. Final draft of presentation
  3. Initial draft of report
  4. Final draft of report
- May
  1. Presentation

## Topic Talkies
* **Data Science for Funsies**: Emphasizing the importance of using data science *for the fun of it*. This encourages the development of data science skills using real, large data sets that are both interesting and relevant to the data scientist themselves. Advantages such as domain knowledge familiarity, increased focus and motivation, the potential for real benefits from results, and opportunities to connect with other people with similar interests (personally or professionally) through self-expression are also explored.

## Linky-Dinks
* Bri's Project:
  * [PyTesseract on PyPI](https://pypi.org/project/pytesseract/)
  * [Tesseract OCR on GitHub](https://github.com/tesseract-ocr/tesseract)
  * [Tesseract's ImproveQuality Documentation](https://github.com/tesseract-ocr/tessdoc/blob/master/ImproveQuality.md)
    * [Rotation Sample from endolith on GitHub](https://gist.github.com/endolith/334196bac1cac45a4893#file-rotation_spacing-py)
    * [Skewing Sample from mzucker on GitHub](https://github.com/mzucker/unproject_text)
    * [OpenCV Image Denoising Tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html)
    * [Canny Edge Detection Step-by-Step on Towards Data Science](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123)
* Data Science Tools:
  * [GitGuide on GitHub](https://github.com/git-guides/)
    * [Creating a Simple Pull Request from Jake Vanderplas on YouTube](https://www.youtube.com/watch?v=rgbCcBNZcdQ)
  * [OriginLab Graphing Software](https://www.originlab.com/index.aspx)
  * [Tableau for Students](https://www.tableau.com/academic/students)
    * [Tableau Tutorial from DSC 201](http://www.cis.umassd.edu/~dkoop/dsc201-2018fa/assignment2.html)
    * [Tableau Tutorial Video by Ben Pfeffer](https://www.youtube.com/watch?v=2sp3HO3Jzfw)
  * [Regex Cheat Sheet for Python](https://www.dataquest.io/wp-content/uploads/2019/03/python-regular-expressions-cheat-sheet.pdf)
  * [R: dplyr Tutorial](https://genomicsclass.github.io/book/pages/dplyr_tutorial.html)
  * [R Programming @ UMassD from Donghui Yan](https://sites.google.com/site/rprogrammingumassd/home)
* UMass Dartmouth Faculty and Groups:
  * [Donghui Yan on UMass Dartmouth](http://www.math.umassd.edu/~dyan/)
    * [Donghui's DSC 101 Page](https://sites.google.com/site/umassddsc101/)
    * [Donghui's MTH 331 Page](https://sites.google.com/site/umassdmth331/)
    * [Donghui's MTH 499/522 Page](https://sites.google.com/site/umassdmth499/)
  * [Scott Field on UMass Dartmouth](http://www.math.umassd.edu/~sfield/)
  * [Alfa Heryudono on UMass Dartmouth](http://www.math.umassd.edu/~aheryudono/)
    * [Alfa's Resources for Talks and Papers](http://www.math.umassd.edu/~aheryudono/mth540f2018site/resources.html)
  * [UMD Big Data Club on GitHub](https://umdbigdataclub.github.io)
  * [(Julia) Hua Fang on UMass Dartmouth](https://www.umassd.edu/directory/hfang2/)
    * [Fang Lab: Computational Statistics and Data Science](https://www.umassmed.edu/fanglab/)
* Guest Speaker and Topic Talk Resources:
  * Guest speakers (check notes for whatever Bri was able to transcribe):
    1. **Apoorva Ramesh** (Feb 17th, Week 5)
    2. **Michelle Barnes** (Mar 3rd, Week 7)
    3. **Hannah Smith** (Apr 7th, Week 11)
    4. **Shawn Manfredo** (Apr 28th, Week 13)
  * [Hack.Diversity](https://www.hackdiversity.com/) - From Michelle Barnes; Boston-based mentoring and career program for PoC in STEM, 
  * Learning Sciences Programs from Hannah Smith:
    1. [ASSISTments](https://new.assistments.org/) - Massive datasets of grade school students' problems and answers
    2. [Wearable Learning Cloud Platform (WLCP)](http://wearablelearning.org/) - Platform for designing and playing math games for teaching
    3. [Graspable Math Activities](https://graspablemath.com/) - Algebra tools for visualizing expression transformations
  * Baseball Analytics Services from Shawn Manfredo:
    1. [HitTrax](https://www.hittrax.com/) - Infrared camera for capturing real-time object trajectory data
    2. [FlightScope](https://flightscope.com/) - Radar and camera hybrid for capturing real-time object trajectory data
    3. [Driveline Baseball](https://www.drivelinebaseball.com/) - Program for training baseball players using data science
  * Topic Talks:
    1. **Brianna**: Robotic Process Automation (RPA) (Feb 24th, Week 6)
       * [Automation Anywhere](https://www.automationanywhere.com/) - Cloud RPA platform, also certification
       * [Lumaxia](https://lumaxia.com/) - RPA training platform
    2. **Marco**: LaTeX, Markdown, and Workflow (Mar 3rd, Week 7)
       * [Overleaf](https://www.overleaf.com/) - Popular online collaborative LaTeX editor
       * [Google Collab Notebooks](https://colab.research.google.com/notebooks/io.ipynb) - Collaborative Python notebooks
    3. **Hunter**: Demographics and the Economy (Mar 17th, Week 8)
       * [One Billion Americans: The Case for Thinking Bigger by Matthew Yglesias](https://www.amazon.com/One-Billion-Americans-Thinking-Bigger-ebook/dp/B082ZR6827) - Part of the discussion
       * [In Russia: Have a baby, Win a Refrigerator from AP](https://www.heraldnet.com/news/in-russia-have-a-baby-win-a-refrigerator/) - Article on the Russian method of promoting population growth
    4. **Brian**: Data Science for Funsies (Mar 24th, Week 9)
       * [How Much Data is Generated Each Day? from World Economic Forum](https://www.weforum.org/agenda/2019/04/how-much-data-is-generated-each-day-cf4bddf29f/) - Quick overview of modern data volumes
       * [So You Want to Make a Hockey Data Viz? by Meghan Hall](https://medium.com/nightingale/so-you-want-to-make-a-hockey-data-viz-dda7b347f117) - Interview on how to approach visualization
       * [Meghan Hall's Tableau Public Page](https://public.tableau.com/profile/meghanhall#!/) - Great examples of visualization
       * [How to train an artificial neural network to play Diablo 2 using visual input?](https://stackoverflow.com/questions/6542274/how-to-train-an-artificial-neural-network-to-play-diablo-2-using-visual-input) - Example of how to apply data science to something neat
       * [Technavio](https://www.technavio.com/) - Data reports on niche topics (e.g. global pancake mix market)
       * [Monster Hunter Rise](https://www.monsterhunter.com/rise/us/) - Monster Hunter is Bri's go-to series for fun data science and finding true love, also Rise is really great
       * [Pokemon on Towards Data Science](https://towardsdatascience.com/tagged/pokemon) - List of articles tagged for Pokemon; lots of examples of how to apply data science to video game data
       * [TrueSkill 2: An improved Bayesian skill rating system](https://www.microsoft.com/en-us/research/uploads/prod/2018/03/trueskill2.pdf) - Matchmaking method used in Gears of War 4's online multiplayer
    5. **Nick**: Three Aspects of Data Fundamentals (Mar 31st, Week 10)
    6. **Jimmy**: Sports and Data Science (Apr 7th, Week 11)
       * [Paul DePodesta on Wikipedia](https://en.wikipedia.org/wiki/Paul_DePodesta) - Sports data scientist turned MLB/NFL executive
       * [James Holzhauer on Wikipedia](https://en.wikipedia.org/wiki/James_Holzhauer) - Jeopardy superstar who uses predictive models for sports gambling
       * [Katerina Wu Has Dream Job as Penguins Data Scientist](https://www.nhl.com/penguins/news/katerina-wu-has-dream-job-as-penguins-data-scientist/c-322941668) - Awesome story of getting a DSC dream job with your favorite pro sports team as a 22-year-old college grad
    7. **Cachelle**: Introduction to Flask & LinkedIn Tips (Apr 14th, Week 12)
       * [Flask Documentation](https://flask.palletsprojects.com/en/1.1.x/)
       * [LinkedIn](https://www.linkedin.com/)
    8. **Ziyu**: Data Visualization
       * [CDC Examples](https://www.cdc.gov/coronavirus/2019-ncov/covid-data/data-visualization.htm)
