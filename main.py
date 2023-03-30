from flask import Flask, render_template, json,request, redirect, url_for, session, flash, make_response
#from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_dance.contrib.google import make_google_blueprint, google
from flask_dance.contrib.facebook import make_facebook_blueprint, facebook
import emailparser as ep
import text_parsing as tp
import webscraping as wb
import os
import datetime
#import new_text as tp
app= Flask(__name__)
app.secret_key="Secret Key"

UPLOAD_FOLDER = '//home/hifza/Pictures/Email/static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf','docx','doc'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI']="mysql+pymysql://root:@localhost/EmailParser"
# app.config['SQLALCHEMY_TRACK_NOTIFICATIONS']=False
db = SQLAlchemy(app)


listt = ["main_content", "headings", "paragraphs", "lists", "list_items", "quotes", "code_blocks",
         "tables", "table_rows", "table_headers", "table_cells", "images", "links", "forms",
         "input_fields", "buttons", "labels", "select_menus", "options", "textareas", "iframes",
         "divs", "sections", "headers", "footers", "navigation_menus", "asides", "figures",
         "captions", "details", "summaries"]

# database content
class Register(db.Model):
    #user_id,user_fullname,user_phno,user_email,user_message
    user_id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(80), nullable=False)
    user_email=db.Column(db.String(80),nullable=False)
    user_password = db.Column(db.String(128), nullable=False)

    def __init__(self,user_name,user_email,user_password):
        self.user_name=user_name
        self.user_email=user_email
        self.user_password = user_password


class TextParsing(db.Model):
    #keywords, regex,text,stopwords, limit,exactmatch, duplicates, direction
    id=db.Column(db.Integer,primary_key=True)
    keywords = db.Column(db.String(80),nullable=False)
    regex = db.Column(db.String(80), nullable=False)
    file = db.Column(db.String(250), nullable=True)
    stopwords=db.Column(db.String(80),nullable=False)
    limit = db.Column(db.String(80), nullable=False)
    exactmatch = db.Column(db.Boolean, nullable=False)
    duplicates = db.Column(db.Boolean, nullable=False)
    direction = db.Column(db.String(80), nullable=False)
    extracted_data = db.Column(db.Text, nullable=False)

    def __init__(self,keywords, regex,file,stopwords, limit,exactmatch, duplicates, direction,extracted_data):
        self.keywords=keywords
        self.regex=regex
        self.file=file
        self.stopwords=stopwords
        self.limit=limit
        self.exactmatch=exactmatch
        self.duplicates=duplicates
        self.direction=direction
        self.extracted_data=extracted_data
#

class EmailParsing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
   # user_id = db.Column(db.Integer, db.ForeignKey('register.user_id'), nullable=False)
    users = db.Column(db.String(80), nullable=False)
    passwords = db.Column(db.String(80), nullable=False)
    msg_from = db.Column(db.String(80), nullable=False)
    value = db.Column(db.String(80), nullable=False)
    keywords = db.Column(db.String(80), nullable=False)
    regex = db.Column(db.String(80), nullable=False)
    stopwords = db.Column(db.String(80), nullable=False)
    limit = db.Column(db.String(80), nullable=False)
    exactmatch = db.Column(db.Boolean, nullable=False)
    duplicates = db.Column(db.Boolean, nullable=False)
    direction = db.Column(db.String(80), nullable=False)
    extracted_data = db.Column(db.Text, nullable=False)

    def __init__(self, users, passwords, msg_from, value, keywords, regex, stopwords, limit, exactmatch, duplicates, direction, extracted_data):
        #self.user_id = user_id
        self.users = users
        self.passwords = passwords
        self.msg_from = msg_from
        self.value = value
        self.keywords = keywords
        self.regex = regex
        self.stopwords = stopwords
        self.limit = limit
        self.exactmatch = exactmatch
        self.duplicates = duplicates
        self.direction = direction
        self.extracted_data = extracted_data


class WebScrape(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    choice = db.Column(db.String(255), nullable=False)
    keywords = db.Column(db.String(255), nullable=False)
    regex = db.Column(db.String(255), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    stopwords = db.Column(db.String(255), nullable=False)
    limit = db.Column(db.Integer, nullable=False)
    exactmatch = db.Column(db.Boolean, nullable=False)
    duplicates = db.Column(db.Boolean, nullable=False)
    direction = db.Column(db.String(10), nullable=False)
    extraced_data=db.Column(db.Text,nullable=False)
    def __init__(self, choice,keywords, regex, stopwords, url,limit, exactmatch, duplicates, direction, extracted_data):
        #self.user_id = user_id
        self.keywords = keywords
        self.regex = regex
        self.url=url
        self.stopwords = stopwords
        self.limit = limit
        self.exactmatch = exactmatch
        self.duplicates = duplicates
        self.direction = direction
        self.extracted_data = extracted_data

# # for creation of tables
with app.app_context():
    # create the database table
    db.create_all()


@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Validate user input
        user_name = request.form['user_name']
        user_email = request.form['user_email']
        user_password = request.form['user_password']

        if not user_name or not user_email or not user_password:
            flash('Please fill out all fields.')
            return redirect(url_for('register'))

        existing_user = Register.query.filter_by(user_name=user_name).first()
        if existing_user:
            flash('That username is already taken.')
            return redirect(url_for('register'))

        # Add the new user to the database
        new_user = Register(user_name=user_name, user_email=user_email, user_password=user_password)
        db.session.add(new_user)
        try:
            db.session.commit()
        except Exception as e:
            flash('An error occurred while creating your account.')
            return redirect(url_for('register'))

        # Create a new session for the user
        session['user_id'] = new_user.user_id
        session['user_name'] = new_user.user_name

        flash('Your account has been created. Please log in.')
        return redirect(url_for('login'))

    # Handle GET requests by returning the registration form
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Validate user input
        user_email = request.form['user_email']
        user_password = request.form['user_password']

        if not user_email or not user_password:
            flash('Please enter your email and password.')
            return redirect(url_for('login'))

        # Check if the user with the provided email exists in the database
        user = Register.query.filter_by(user_email=user_email).first()
        if not user:
            flash('Incorrect email or password.')
            return redirect(url_for('login'))

        # Verify the password
        if user.user_password != user_password:
            flash('Incorrect email or password.')
            return redirect(url_for('login'))

        # Store the user ID and username in the session
        session['user_id'] = user.user_id
        session['user_name'] = user.user_name

        flash('You have been logged in.')
        return redirect(url_for('home'))

    # Handle GET requests by returning the login form
    return render_template('login.html')


@app.route('/forgot')
def forgot():
    return render_template('forgot-password.html')


@app.route('/utilities')
def utlities():
    return render_template('doc_output.html')

@app.route('/about_document')
def about_document():
    return render_template('about_document.html')
@app.route('/about_email')
def about_email():
    return render_template('about_email.html')

@app.route('/error_page')
def error():
    return render_template('404.html')

listt = ["main_content", "headings", "paragraphs", "lists", "list_items", "quotes", "code_blocks",
         "tables", "table_rows", "table_headers", "table_cells", "images", "links", "forms",
         "input_fields", "buttons", "labels", "select_menus", "options", "textareas", "iframes",
         "divs", "sections", "headers", "footers", "navigation_menus", "asides", "figures",
         "captions", "details", "summaries"]
# @app.route('/webscrape', methods=['GET','POST'])
# def webscrape():
#
#     if request.method=="POST":
#
#         #[int(c.split('option')[-1]) for c in
#         choices=request.form.getlist('choices')
#         if not choices:
#             os.abort(400, 'Please select at least one option')
#         keywords = request.form['keywords']
#         regex = request.form['regex']
#         url = request.form['url']
#         stopwords = request.form['proximity_stop_words']
#         limit = request.form['limit']
#         exactmatch = request.form['exactmatch']
#         duplicates = request.form['duplicates']
#         direction = request.form['direction']
#        # data=keywords,regex,stopwords,limit,exactmatch,duplicates,direction
#         extracted_data = wb.document_extraction(choices,keywords, regex,url,stopwords, limit,exactmatch, duplicates, direction)
#         #return extracted_data
#         # Perform web scraping and data extraction
#
#         return render_template('scrape_output.html', res=extracted_data)
#     return render_template('webscraping.html', listt=listt)
#
#



@app.route('/webscrape', methods=['GET', 'POST'])
def webscrape():
    if request.method == 'POST':

        # Retrieve the form data
        choices = request.form.getlist('choices')
        keywords = request.form['keywords']
        regex = request.form['regex']
        url = request.form['url']
        stopwords = request.form['proximity_stop_words']
        limit = int(request.form['limit'])
        exactmatch = bool(request.form.get('exactmatch'))
        duplicates = bool(request.form.get('duplicates'))
        direction = request.form['direction']
        # Perform web scraping and data extraction
        extracted_data = wb.document_extraction(choices, keywords, regex, url, stopwords, limit, exactmatch, duplicates,
                                                direction)
       # Save the form data to the database
        form_data = WebScrape(choice=choices,keywords=keywords, regex=regex, url=url, stopwords=stopwords, limit=limit,
                             exactmatch=exactmatch, duplicates=duplicates, direction=direction,extracted_data=extracted_data)
        db.session.add(form_data)
        try:
            db.session.commit()
        except Exception as e:
            flash('An error occurred while entering the data.')
        return render_template('scrape_output.html', res=extracted_data)
    return render_template('webscraping.html', listt=listt)


@app.route('/check_email', methods=['GET','POST'])
def check_email():
    if request.method == "POST":
        # (user: str, password: str, msg_from: str, value: str, keyword: str, regex: str,
        #                      proximity_stop_words: str, limit, exact_match: bool, duplicates: bool, direction: str):
        user = request.form['user']
        passwords = request.form['password']  # wyashvhufbssddga
        msg_from = request.form['msg_from']
        value = request.form['value']
        keywords = request.form['keyword']
        regex = request.form['regex']
        stopwords = request.form['proximity_stop_words']
        limit = request.form['limit']
        exactmatch = request.form['exact_match']
        exactmatch = True if request.form.get('exact_match') == 'True' else False
        duplicates = True if request.form.get('duplicates') == 'True' else False
        # duplicates = request.form['duplicates']
        direction = request.form['direction']
        # data=keywords,regex,stopwords,limit,exactmatch,duplicates,direction
        extracted_data = ep.email_extraction(user, passwords, msg_from, value, keywords, regex, stopwords, limit,
                                             exactmatch, duplicates,
                                            direction)

        Extracted_Email = EmailParsing(users=user, passwords=passwords, msg_from=msg_from,value=value,
                                   keywords=keywords,regex=regex,stopwords=stopwords,limit=limit, exactmatch=exactmatch,
                                   duplicates=duplicates,direction=direction,extracted_data=extracted_data)
        db.session.add(Extracted_Email)
        try:
            db.session.commit()
        except Exception as e:
            flash('An error occurred while entering the data.')
        return render_template('email_output.html', message=extracted_data)
    return render_template('emailparser.html')

@app.route('/')
def charts():
    return render_template('index2.html')


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'doc', 'docx'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/check_document', methods=['GET', 'POST'])
def check_document():
    if request.method == 'POST':
        keywords = request.form['keywords']
        regex = request.form['regex']
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        file_name=secure_filename(file.filename)
        if file.filename == '':
            print('No file selected for uploading')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        stopwords = request.form['proximity_stop_words']
        limit = request.form['limit']
        exactmatch = True if request.form.get('exact_match') == 'True' else False
        duplicates = True if request.form.get('duplicates') == 'True' else False
        direction = request.form['direction']
        # Pass the file path to the document_extraction function
        extracted_data = tp.document_extraction(keywords, regex,
                                                os.path.join(app.config['UPLOAD_FOLDER'], filename), stopwords,
                                                limit, exactmatch, duplicates, direction)
        Extracted_doc = TextParsing(keywords=keywords, regex=regex, file=file_name, stopwords=stopwords,
                                        limit=limit,exactmatch=exactmatch,duplicates=duplicates, direction=direction, extracted_data=extracted_data)
        db.session.add(Extracted_doc)
        db.session.commit()
        return render_template('doc_output.html', res=extracted_data)
    return render_template('documentparser.html')




if __name__=="__main__":
    app.run(debug=True,port=3000)