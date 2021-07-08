import pandas as pd
import numpy as np
import requests
import time
import os
from tqdm import tqdm
from pyarrow import feather


def get_recent_changes(N):
    S = requests.Session()

    t = tqdm(total=N, position=0, leave=True)

    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "format": "json",
        "rcprop": "title|ids|sizes|flags|user|userid|timestamp",
        "rcshow": "!bot|!anon|!minor",
        "rctype": "edit",
        "rcnamespace": "0",
        "list": "recentchanges",
        "action": "query",
        "rclimit": "500",
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    RECENTCHANGES = DATA["query"]["recentchanges"]
    all_rc = RECENTCHANGES

    i = 500
    t.update(500)
    while i <= N:
        last_continue = DATA["continue"]
        PARAMS.update(last_continue)
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        RECENTCHANGES = DATA["query"]["recentchanges"]
        all_rc.extend(RECENTCHANGES)
        i = i + 500
        t.update(500)

    if len(all_rc) > N:
        all_rc = all_rc[:N]

    return all_rc


def get_sample_of_users(edit_lookback, outfile=None):
    """Get a sample of recently active users by pulling the most recent N edits
    Note that this will be biased towards highly active users.
    Args:
        edit_lookback: The number of edits to go back.
        outfile: Pickle file path to write the user list to
    Returns:
        Dataframe with user and user id columns
    """
    df = get_recent_changes(edit_lookback)

    # Drop missing userid entries
    df = pd.DataFrame(df).dropna(subset=["userid"])

    print("Earliest timestamp: {}".format(df.timestamp.min()))
    print("Latest timestamp: {}".format(df.timestamp.max()))
    print("Number of distinct users: {}".format(len(df.user.unique())))
    print(
        "Mean number of edits per user in timeframe: %.2f"
        % (len(df) / len(df.user.unique()))
    )
    print("Number of distinct pages edited: {}".format(len(df.pageid.unique())))
    print(
        "Mean number of edits per page in timeframe: %.2f"
        % (len(df) / len(df.pageid.unique()))
    )

    # Deduplicate to get
    sampled_users = df.loc[:, ["user", "userid"]].drop_duplicates()

    # Remove RFD
    sampled_users = sampled_users[np.invert(sampled_users.user == "RFD")]
    sampled_users = sampled_users.reset_index(drop=True)

    if outfile:
        sampled_users.to_csv(outfile, index=False)

    return sampled_users


def get_edit_history(
    userid=None, user=None, latest_timestamp=None, earliest_timestamp=None, limit=None):
    """For a particular user, pull their whole history of edits.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    S = requests.Session()
    S.headers.update(
        {"User-Agent": "WikiRecs (danielrsaunders@gmail.com) One-time pull"}
    )

    URL = "https://en.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "format": "json",
        "ucnamespace": "0",
        "list": "usercontribs",
        "ucuserids": userid,
        "ucprop": "title|ids|sizediff|flags|comment|timestamp",
        "ucshow=": "!minor|!new",
    }
    if latest_timestamp is not None:
        PARAMS["ucstart"] = latest_timestamp
    if earliest_timestamp is not None:
        PARAMS["ucend"] = earliest_timestamp
    if user is not None:
        PARAMS["ucuser"] = user
    if userid is not None:
        PARAMS["ucuserid"] = userid

    PARAMS["uclimit"] = 500

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    if "query" not in DATA:
        print(DATA)
        raise ValueError

    USERCONTRIBS = DATA["query"]["usercontribs"]
    all_ucs = USERCONTRIBS
    i = 500
    while i < 100000:
        if "continue" not in DATA:
            break
        last_continue = DATA["continue"]
        PARAMS.update(last_continue)
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        USERCONTRIBS = DATA["query"]["usercontribs"]
        all_ucs.extend(USERCONTRIBS)
        i = i + 500

    return all_ucs


def pull_edit_histories(
    sampled_users_file,
    edit_histories_file_pattern,
    users_per_chunk,
    earliest_timestamp,
    start=0):
    histories = []
    cols = ["userid", "user", "pageid", "title", "timestamp", "sizediff"]
    sampled_users = pd.read_csv(sampled_users_file)
    sampled_users.loc[:, "userid"].astype(int)

    sampled_users = sampled_users.reset_index()

    for i in range(len(sampled_users)//users_per_chunk):
      if os.path.exists(edit_histories_file_pattern.format(i*users_per_chunk)):
        # print(edit_histories_file_pattern.format(i*users_per_chunk))
        start=((i+1)*users_per_chunk)
      else:
        break
        
    print("Starting from {}".format(start))

    # Iterate through all the users in the list
    for i, (user, userid) in tqdm(
        iterable=enumerate(
            zip(sampled_users["user"][start:], sampled_users["userid"][start:]),
            start=start),
        total=len(sampled_users)-start,
        position=0, leave=True): 

        # Get the history of edits for this userid
        thehistory = get_edit_history(
            userid=int(userid), earliest_timestamp=earliest_timestamp
        )

        # If no edits, skip
        if len(thehistory) == 0:
            continue

        thehistory = pd.DataFrame(thehistory)

        # Remove edits using automated tools by looking for the word "using" in the comments
        try:
            thehistory = thehistory[
                np.invert(thehistory.comment.astype(str).str.contains("using"))
            ]
        except AttributeError:
            continue

        if len(thehistory) == 0:
            continue

        histories.append(thehistory.loc[:, cols])

        # if np.mod(i, 50) == 0:
        #     print(
        #         "Most recent: {}/{} {} ({}) has {} edits".format(
        #             i, len(sampled_users), user, int(userid), len(thehistory)
        #         )
        #     )

        # Every x users save it out, for the sake of ram limitations
        if np.mod(i, users_per_chunk) == 0:
            feather.write_feather(
                pd.concat(histories), edit_histories_file_pattern.format(i)
            )

            histories = []
      
    # Get the last few users that don't make up a full chunk
    feather.write_feather(pd.concat(histories), edit_histories_file_pattern.format(i))