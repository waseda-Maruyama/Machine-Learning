from datetime import datetime
from dateutil import tz
import jquantsapi

my_mail_address:str = "*****"
my_password: str = "*****"
cli = jquantsapi.Client(mail_address=my_mail_address, password=my_password)
df = cli.get_price_range(
    start_dt=datetime(2022, 7, 25, tzinfo=tz.gettz("Asia/Tokyo")),
    end_dt=datetime(2022, 7, 26, tzinfo=tz.gettz("Asia/Tokyo")),
)
print(df)