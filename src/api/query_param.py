from datetime import datetime, timedelta
import pytz

def get_query_param()->dict:
    #Get the yesterday date value
    utc_time = datetime.now(pytz.utc)
    utc_date = utc_time.date()
    yesterday = utc_date - timedelta(days=1)

    query_params = {
        "startDate": yesterday,
        "endDate": yesterday
    }

    return query_params

