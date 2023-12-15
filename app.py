import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime
import statsmodels

top_nba_players = [
    "Michael Jordan","LeBron James","Kobe Bryant","Magic Johnson","Larry Bird","Shaquille O'Neal","Tim Duncan",
    "Hakeem Olajuwon","Kareem Abdul-Jabbar","Wilt Chamberlain","Bill Russell","Oscar Robertson",
    "Jerry West","Elgin Baylor","David Robinson",
    "Kevin Garnett","Dirk Nowitzki","Karl Malone","John Stockton","Dwyane Wade","Scottie Pippen","Charles Barkley",
    "Allen Iverson","Isiah Thomas","Stephen Curry","Kawhi Leonard","Giannis Antetokounmpo",
    "Kevin Durant","Chris Paul","Paul Pierce","Clyde Drexler","Patrick Ewing","John Havlicek","Dominique Wilkins",
    "Tracy McGrady","George Gervin","Manu Ginobili","Russell Westbrook","James Harden","Chris Bosh","Ray Allen",
    "Pau Gasol","Tony Parker","Chris Webber","Grant Hill","Yao Ming","Steve Nash","Reggie Miller","Nate Thurmond"
]

# Load Player Data
player_data = pd.read_csv("archive/Player Per Game.csv")
player_data = player_data[player_data["lg"]=="NBA"]
# in player data team is tm
team_data = pd.read_csv("archive/Team Summaries.csv")
team_data = team_data[team_data["lg"]=="NBA"]
# in team data team is abbreviation
# advanced data
adv_data = pd.read_csv("archive/Advanced.csv")
adv_data = adv_data[adv_data["lg"]=="NBA"]

# all star voting
teamVoting = pd.read_csv("archive/End of Season Teams (Voting).csv")
teamVoting = teamVoting[teamVoting["lg"]=="NBA"]
teamVoting = teamVoting.loc[teamVoting["season"] != 2024]

# frankensteins baby
playerdata = pd.merge(player_data, adv_data, on=['seas_id', 'player_id', 'player', 'season', 'birth_year',
                                                 'pos', 'age', 'experience', 'tm', 'lg', 'g'], suffixes=('_left','_right'))
playerdata = playerdata.loc[playerdata['season'] != 2024]
season_years = playerdata['season'].unique()

#playerdata = playerdata.rename(columns={'pts_per_game':'Points Per Game','ws':'Win Shares','mp_per_game':'Minutes Per Game'
#                                        ,'ts_percent':'True Shooting %','usg_percent':'Usage Rate',
#                                        'g':'Games Played','w':'Games Won','per':'Player Efficiency Rating'})

columns_to_delete = [
  "seas_id",
  "season",
  "player_id",
  "player",
  "birth_year",
  "pos",
  "age",
  "experience",
  "lg",
  "tm",
]
def change_column_names(dataframe):
    # Define a mapping dictionary for column renaming
    column_mapping = {
        "g": "Games Played",
        "gs": "Games Started",
        "mp_per_game": "Minutes Per Game",
        "fg_per_game": "Field Goals Made Per Game",
        "fga_per_game": "Field Goals Attempted Per Game",
        "fg_percent": "Field Goal Percentage",
        "x3p_per_game": "Three-Pointers Made Per Game",
        "x3pa_per_game": "Three-Pointers Attempted Per Game",
        "x3p_percent": "Three-Point Percentage",
        "x2p_per_game": "Two-Pointers Made Per Game",
        "x2pa_per_game": "Two-Pointers Attempted Per Game",
        "x2p_percent": "Two-Point Percentage",
        "e_fg_percent": "Effective Field Goal Percentage",
        "ft_per_game": "Free Throws Made Per Game",
        "fta_per_game": "Free Throws Attempted Per Game",
        "ft_percent": "Free Throw Percentage",
        "orb_per_game": "Offensive Rebounds Per Game",
        "drb_per_game": "Defensive Rebounds Per Game",
        "trb_per_game": "Total Rebounds Per Game",
        "ast_per_game": "Assists Per Game",
        "stl_per_game": "Steals Per Game",
        "blk_per_game": "Blocks Per Game",
        "tov_per_game": "Turnovers Per Game",
        "pf_per_game": "Personal Fouls Per Game",
        "pts_per_game": "Points Per Game",
        "mp": "Minutes Played",
        "per": "Player Efficiency Rating",
        "ts_percent": "True Shooting Percentage",
        "x3p_ar": "Three-Point Attempt Rate",
        "f_tr": "Free Throw Rate",
        "orb_percent": "Offensive Rebound Percentage",
        "drb_percent": "Defensive Rebound Percentage",
        "trb_percent": "Total Rebound Percentage",
        "ast_percent": "Assist Percentage",
        "stl_percent": "Steal Percentage",
        "blk_percent": "Block Percentage",
        "tov_percent": "Turnover Percentage",
        "usg_percent": "Usage Rate",
        "ows": "Offensive Win Shares",
        "dws": "Defensive Win Shares",
        "ws": "Total Win Shares",
        "ws_48": "Win Shares per 48 Minutes",
        "obpm": "Offensive Box Plus-Minus",
        "dbpm": "Defensive Box Plus-Minus",
        "bpm": "Box Plus-Minus",
        "vorp": "Value Over Replacement Player"
    }

    # Rename columns of the DataFrame based on the mapping dictionary
    dataframe.columns = [column_mapping.get(col, col) for col in dataframe.columns]




change_column_names(playerdata)
stats = playerdata.columns.tolist()
stats = [col for col in stats if col not in columns_to_delete]
#st.write(stats)


#playerVTeamStats = {'Points Per Game','Win Shares','Minutes Per Game','True Shooting %','Usage Rate', 'Player Efficiency Rating'}
playerVTeamCompare = {'Games Won','Games Played'}
# get star player function
def get_star_players(player_data, team_data, season, games_played_threshold=30):
    star_players = set()
    star_players_rows = []
    
    # Get team abbreviations for the specific season
    team_year_data = team_data[team_data["season"] == season]
    player_year_data = player_data[player_data["season"] == season]

    temp_team_abr = team_year_data["abbreviation"].unique()
    # Get rid of any nan
    team_abr = [i for i in temp_team_abr if i is not np.nan]

    for team in team_abr:
        cur_data = player_year_data.loc[player_year_data['tm'] == team]

        # Add condition to check the number of games played
        cur_data = cur_data[cur_data['Games Played'] > games_played_threshold]

        if not cur_data.empty:
            # Find the index of the player with the maximum points per game
            max_pts_index = cur_data['Points Per Game'].idxmax()
            star_players_rows.append(cur_data.loc[max_pts_index])
            
            max_pts_player_name = cur_data.loc[max_pts_index, 'player']
            
            # Check if the player's name is not already in the set
            if max_pts_player_name not in star_players:
                star_players.add(max_pts_player_name)
    star_players_df = pd.DataFrame(star_players_rows)
    star_players_df['Games Won']=0

    for index, row in star_players_df.iterrows():
        team_info = team_data[(team_data['abbreviation'] == row['tm']) & (team_data['season'] == row['season'])]
    
        # If the team is found, update the 'wins' column with the corresponding value
        if not team_info.empty:
            star_players_df.at[index, 'Games Won'] = team_info['w'].values[0]
    
    return star_players_df


ErasStats = {"Avg True Shooting %","Players Avging Over 25", "Avg 3-pt Attempts Per Game", "Fouls Per Game", "FT Per Game"}
def decades_basketball(playerdata, daterange):
    start, end = daterange
    season_range = playerdata.loc[(playerdata['season'] >= start) & (playerdata['season'] <= end)]

    # Calculate mean ts_percent and average x3pa_per_game for each season
    mean_ts_pct = season_range.groupby('season')['True Shooting Percentage'].mean().reset_index(name='Avg True Shooting %')
    avg_x3pa_per_game = season_range.groupby('season')['Three-Pointers Attempted Per Game'].mean().reset_index(name='Avg 3-pt Attempts Per Game')
    avg_pf_per_game = season_range.groupby('season')['Personal Fouls Per Game'].mean().reset_index(name="Fouls Per Game")
    fta_per_game = season_range.groupby('season')['Free Throws Attempted Per Game'].mean().reset_index(name="FT Per Game")

    # Merge the three DataFrames on 'season'
    
    decades = pd.merge(mean_ts_pct, avg_x3pa_per_game, on='season')
    decades = pd.merge(decades, fta_per_game, on='season')
    decades = pd.merge(decades, avg_pf_per_game, on='season')
    # Filter players with pts_per_game higher than 25 and count them for each season
    filtered_players = season_range[season_range['Points Per Game'] > 25]
    player_count = filtered_players.groupby('season')['player'].count().reset_index(name='Players Avging Over 25')
    decades = decades.fillna(0)
    # Merge the two DataFrames on 'season'
    decades = pd.merge(decades, player_count, on='season')

    return decades


def team_average_stats(playerdata, season):
    for index, row in playerdata.iterrows():
        team_info = team_data[(team_data['abbreviation'] == row['tm']) & (team_data['season'] == row['season'])]
    
        # If the team is found, update the 'wins' column with the corresponding value
        if not team_info.empty:
            playerdata.at[index, 'Games Won'] = team_info['w'].values[0]
    # Select only numeric columns for team averages
    numeric_columns = playerdata[playerdata['season']==season].select_dtypes(include=['number']).columns
    # Print information about selected numeric columns
    print("Selected Numeric Columns:", numeric_columns)

    # Fill missing values with 0 before calculating the mean
    playerdata_filled = playerdata.fillna(0)

    # Group by team and calculate the mean for each numeric column
    team_avg_stats = playerdata_filled.groupby(['tm'])[numeric_columns].mean()

    team_avg_stats = team_avg_stats.reset_index().drop_duplicates(subset=['tm'])
    # Print information about the resulting DataFrame
    print("Team Average Stats DataFrame:")
    print(team_avg_stats)

    return team_avg_stats


# actual stuff!
st.title("Champion Makers: Exploring the Role of NBA Superstars!")
st.subheader("By: Henry Baer")
st.markdown('---')

PVT, BC, ODE, POT, PC, SVT = st.tabs(["Player vs Team Performance", "Player Distributions", "Players Over the Eras", "Player Over Time",
                                  "Player Comparisons", "Star vs Team Avg"])
with PVT:
    st.header("Star Player Relation to Team Performance")
    st.write("This scatter chart attempts to show how different player statistics might have had a role in how well that players ",
            "team may have performed.")
    # column layout
    TvPcol1, TvPcol2 = st.columns(2)
    #st.write(playerdata)
    # option choice
    year = TvPcol1.selectbox("What NBA Season for Player vs Team Performance?", season_years)
    selected_y = TvPcol2.selectbox("What Stat for Player vs Team Performance on Y-Axis", stats)
    #selected_x = TvPcol3.selectbox("What Stat for Player vs Team Performance on X-Axis", playerVTeamCompare)
    sel_data = get_star_players(playerdata, team_data, year)
    if st.checkbox("Show Raw NBA Star Data for Selected Year"):
        st.subheader('Raw Data')
        st.write(sel_data)
    
    #if selected
    fig = px.scatter(sel_data, x="Games Won", y=selected_y, hover_name="player", text="player", trendline='ols')
    fig.update_traces(textposition="top center")
    fig.update_layout(
        title="Player vs Team Performance",
        xaxis_title = "Games Won",
        yaxis_title = selected_y,
        #plot_bgcolor = "white",
        font=dict(color="white")
    )
    st.plotly_chart(fig)
    st.write("This chart also contains a trend line to help visualize if a trend is actually there or not.")
with ODE:
    st.header("How Players Have Evolved Over the Years")
    st.write("This bubble plot displays a couple different type of statistics for specific years. It then plots those to show",
            "any form of trend that might have come over those coming season.")
    ODEcol1, ODEcol2 = st.columns(2)
    ErasRange = ODEcol1.slider("What Season Range?", 1950, 2023, (2010, 2015))
    selected_y2 = ODEcol2.selectbox("What Stat for Player Eras on Y-Axis", ErasStats)
    sel_data = decades_basketball(playerdata, ErasRange)
    if st.checkbox("Animated View?"):
        fig2 = px.scatter(sel_data, x=sel_data['season'], y=selected_y2, size=selected_y2, text="season", color=selected_y2,
                         color_continuous_scale='Plasma', animation_frame="season")
        fig2.update_layout(
            title="Player Stats Over Time",
            xaxis_title = "Season",
            yaxis_title = selected_y2,
            font=dict(color="white")
        )
        fig2.update_xaxes(range=[sel_data["season"].min(), sel_data["season"].max()])
        fig2.update_yaxes(range=[0, sel_data[selected_y2].max()])
    else:
        fig2 = px.scatter(sel_data, x=sel_data['season'], y=selected_y2, size=selected_y2, text="season", color=selected_y2,
                         color_continuous_scale='Plasma')
    st.plotly_chart(fig2)
    st.write("There is also an animation button that upon clicking will allow the user to look at an animation of the chart!")
with POT:
    st.header("What about Individual Player Stats?")
    st.write("What can we learn about a players career when looking at their stats over their career?")
    POTcol1, POTcol2 = st.columns(2)
    player_selection = POTcol1.multiselect("Select Players: ", top_nba_players)
    selected_y3 = POTcol2.selectbox("What Stat To Track", stats)
    selected_players_data = playerdata[playerdata['player'].isin(player_selection)]
    fig3 = px.line(selected_players_data, x="season", y=selected_y3, markers=True, color="player")
    fig3.update_layout(
            title="Player Stats Over Time",
            xaxis_title = "Season",
            yaxis_title = selected_y3,
            font=dict(color="white")
        )
    st.plotly_chart(fig3)
with PC:
    st.header("Player Stat Correlation")
    st.write("How different players compare to league averages. This may help illustrate how",
             "a player ranks among their colleagues and if the stats are correlated or not.")
    PCcol2, PCcol3 = st.columns(2)
    year3 = st.selectbox("What NBA Season for Player Comparisons?", season_years)
    PCselected_x = PCcol2.selectbox("Stat to track on X-Axis", stats)
    PCselected_y = PCcol3.selectbox("Stat to track on Y-Axis", stats)
    compare_data = playerdata[playerdata['season']==year3]

    x_avg = compare_data[PCselected_x].mean()
    y_avg = compare_data[PCselected_y].mean()

    if st.checkbox("Add trendline?"):
        fig4 = px.scatter(compare_data, x=PCselected_x, y=PCselected_y, hover_name="player", color="tm", trendline="ols",
                          trendline_scope="overall")

    else:
       fig4 = px.scatter(compare_data, x=PCselected_x, y=PCselected_y, hover_name="player", color="tm")

    fig4.update_layout(
            title="Player Comparisons",
            xaxis_title = PCselected_x,
            yaxis_title = PCselected_y,
            font=dict(color="white")
    )
    fig4.add_shape(
        go.layout.Shape(
            type="line",
            x0=x_avg,
            x1=x_avg,
            y0=compare_data[PCselected_y].min(),
            y1=compare_data[PCselected_y].max(),
            line=dict(color="red", width=2, dash="dash"),
        )
    )
    fig4.add_annotation(
        go.layout.Annotation(
            x=x_avg,
            y=(compare_data[PCselected_y].min() + compare_data[PCselected_y].max()) / 2,
            text=f'X-Avg: {x_avg:.2f}',
            showarrow=False,
            ax=-50,  # Adjust this value for positioning the annotation on the left side
            ay=0,#compare_data[PCselected_y].max(),
            xshift= 40,
            yshift=100,  # Adjust this value for fine-tuning the position
        )
    )
    fig4.add_shape(
        go.layout.Shape(
            type="line",
            x0=compare_data[PCselected_x].min(),
            x1=compare_data[PCselected_x].max(),
            y0=y_avg,
            y1=y_avg,
            line=dict(color="green", width=2, dash="dash"),
        )
    )
    fig4.add_annotation(
        go.layout.Annotation(
            x=(compare_data[PCselected_x].min() + compare_data[PCselected_x].max()) / 2,
            y=y_avg,
            text=f'Y-Avg: {y_avg:.2f}',
            showarrow=False,
            ax=0,
            ay=50,  # Adjust this value for positioning the annotation above the line
            yshift=20,  # Adjust this value for fine-tuning the position
            xshift=250
        )
    )
    st.plotly_chart(fig4)

with SVT:
    st.header("Differences between star player averages vs their team averages")
    st.write("This chart compares star players vs their team averages to help visualize on what might have a larger impact,",
            "the star player perfomance or their supporting cast.")
    year4 = st.selectbox("What NBA Season for Player vs Team Avgs?", season_years)
    compare_data = playerdata[playerdata['season']==year4]
    star_players = get_star_players(compare_data, team_data, year4)
    result_df = team_average_stats(compare_data, year4)
    SVT2, SVT3 = st.columns(2)
    #SVTselected_x = SVT1.selectbox("Stat track on X-Axis", stats)
    SVTselected_y = SVT2.selectbox("Stat track on Y-Axis", stats)
    SVTsize = SVT3.selectbox("Size of bubble", stats)
    
    y_avg = compare_data[SVTselected_y].mean()

    star_players = star_players.sort_values(by='tm')
    
    fig5 = px.scatter(star_players, x="Games Won", y=SVTselected_y,
                     size=SVTsize, color='tm',  hover_name='player',trendline="ols", trendline_scope="overall",
                     title='Star Players')
    
    fig6 = px.scatter(result_df, x="Games Won", y=SVTselected_y,
                     size=SVTsize, color='tm', hover_name='tm', trendline="ols", trendline_scope="overall",
                     title='Team Averages')
    
    # Display the plot with Streamlit
    st.plotly_chart(fig5)
    st.plotly_chart(fig6)
with BC:
    st.header("Player Distribution for Specific Statistics")
    st.write("To help show the distribution of players and how star players might stand out among",
            "average players")
    BC, BC2 = st.columns(2)
    year5 = BC.selectbox("What NBA season for histogram?", season_years)
    BCselected_y = BC2.selectbox("Stat for histogram", stats)
    new_data = playerdata[playerdata['season']==year5]
    fig7 = px.histogram(new_data, x=BCselected_y)
    st.plotly_chart(fig7)
