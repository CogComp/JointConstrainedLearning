import smtplib
import socket

# Reference:
# https://towardsdatascience.com/notify-with-python-41b77d51657e
# https://stackoverflow.com/questions/16512592/login-credentials-not-working-with-gmail-smtp
def send(msg, server="smtp.gmail.com", port=587):
    # contain following in try-except in case of momentary network errors
    try:
        # initialise connection to email server
        smtp = smtplib.SMTP("smtp.gmail.com", 587)
        # this is the 'Extended Hello' command, essentially greeting our SMTP or ESMTP server
        smtp.ehlo()
        # this is the 'Start Transport Layer Security' command, tells the server we will
        # be communicating with TLS encryption
        smtp.starttls()
        smtp.ehlo()
        
        email = "notifyme616@gmail.com"
        pwd = "616notifyme"
        # login to server
        smtp.login(email, pwd)
        # send notification to self
        smtp.sendmail(email, "why16gzl@seas.upenn.edu", msg.as_string())
        # disconnect from the server
        smtp.quit()
    except socket.gaierror:
        print("Network connection error, email not sent.")