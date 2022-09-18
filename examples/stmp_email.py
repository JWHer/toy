# import smtplib
# #SERVER = "localhost"

# FROM = 'autocare_tx@noreply.com'

# TO = ["jwher@snuailab.ai"] # must be a list

# SUBJECT = "Hello!"

# TEXT = "This message was sent with Python's smtplib."

# # Prepare actual message

# message = """\
# From: %s
# To: %s
# Subject: %s

# %s
# """ % (FROM, ", ".join(TO), SUBJECT, TEXT)

# # Send the mail

# try:
#     server = smtplib.SMTP('localhost')
#     server.sendmail(FROM, TO, message)
#     print("Successfully sent email")
# except Exception as e:
#     print("Error: unable to send email")
#     print(e)
# finally:
#     server.quit()

# import smtplib

# sender = 'hacell2@gmail.com'
# receivers = ['jwher@snuailab.ai']
# receiver_names = [ name.split('@')[0] for name in receivers ]

# message = f"""From: Autocare TX <autocare_tx@noreply.com>
# To: {receiver_names[0]} <jwher@snuailab.ai>
# Subject: Second SMTP e-mail test

# This is a test e-mail message.
# How awesome is it!
# """

# try:
#    smtpObj = smtplib.SMTP('smtp.gmail.com', 587)
#    smtpObj.ehlo()
#    smtpObj.starttls()
#    smtpObj.ehlo()
#    smtpObj.login(sender, '')
#    smtpObj.sendmail(sender, receivers, message)         
#    print("Successfully sent email")
# except Exception as e:
#    print("Error: unable to send email")
#    print(e)
# finally:
#     smtpObj.quit()

import smtplib

sender = 'jwher@noreply.com'
receivers = ['jeongho.kim@noreply.com']
receiver_names = [ name.split('@')[0] for name in receivers ]
subject = 'First internal mail'

message = f"""From: {sender.split('@')[0]} <{sender}>
To: {receiver_names[0]} <{receivers[0]}>
Subject: {subject}

This is a test e-mail message.
"""

try:
   smtpObj = smtplib.SMTP('localhost', 587)
   smtpObj.ehlo()
   smtpObj.starttls()
   smtpObj.ehlo()
   smtpObj.login(sender, '3971')
   smtpObj.sendmail(sender, receivers, message)         
   print("Successfully sent email")
except Exception as e:
   print("Error: unable to send email")
   print(e)
finally:
    smtpObj.quit()