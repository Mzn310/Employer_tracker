from dotenv import load_dotenv
load_dotenv()

from employee_tracking_fixed import EmployeeTracker
import time

t = EmployeeTracker()
print("SMTP user loaded:", t.smtp_user)          # sanity check env vars loaded
print("Alert recipient:", t.alert_recipient)

t.absence_threshold = 5
t.send_absence_alert(12.3)

time.sleep(3)  # let the background thread finish