import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import seaborn as sns
import datetime as dt
import plotly.express as px
from datetime import datetime


# Load the dataset
dataset=pd.read_csv('Playlist.csv',encoding = 'unicode_escape', engine ='python')


def home_section(dataset):
    dataset_container = st.container()

    with dataset_container:
        st.markdown('<h1 style="text-align: center; color: #1DB954;">Welcome to Spotify Data Analysis Website</h1>', unsafe_allow_html=True)
        
        SpotifySongID = pd.DataFrame(dataset["Spotify ID"].value_counts()).head(20)
        
        # Customizing the bar chart
        fig = go.Figure(data=[go.Bar(x=SpotifySongID.index, y=SpotifySongID["Spotify ID"], marker_color="#1DB954")])
        
        fig.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#f8f9fa",
            font=dict(color="#333333"),
            margin=dict(t=60),
        )
        
        fig.update_traces(marker_line_color="#ffffff", marker_line_width=1.5)
        st.plotly_chart(fig)
        st.markdown('''
        <div style="font-family: 'Arial'; font-size: 18px; line-height: 1.6; color: #333333;">
            This Spotify Data Analysis app allows you to explore and analyze Spotify playlist data. You can perform various analyses, such as EDA (Exploratory Data Analysis), trend analysis, and user behavior analysis. Upload your CSV file containing the Spotify playlist data and dive into the insights!
            <br><br>
            <strong>How to use:</strong>
            <ol>
                <li>Select an option from the sidebar to navigate to different sections of the app.</li>
                <li>In the "EDA" section, you can explore the dataset, view summary statistics, correlations, and visualize the data.</li>
                <li>The "Trend Analysis" section provides visualizations of trends in artist popularity, genres, track duration, and the number of tracks over the years.</li>
                <li>The "User Behavior Analysis" section allows you to upload your own user data CSV file and perform analysis on the most played song, most loved genre, and additional insights.</li>
            </ol>
            <br>
            Enjoy exploring and analyzing your Spotify playlist data with this app!
        </div>
        ''', unsafe_allow_html=True)

def perform_eda(dataset):
    
    st.markdown('<h2 style="text-align: center; color: #1DB954;">EXPLORATORY DATA ANALYSIS</h2>', unsafe_allow_html=True)
    dataset_container = st.container()
    with dataset_container:
        st.subheader("EDA")
        st.dataframe(dataset.head(5))
        
        if st.checkbox("Show Summary of Dataset"):
            st.write(dataset.describe())
        
        if st.checkbox("Select Columns To Show"):
            all_columns = dataset.columns.tolist()
            selected_columns = st.multiselect('Select', all_columns)
            new_df = dataset[selected_columns]
            st.dataframe(new_df)
        
        if st.checkbox("Show Correlation Plot"):
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(sns.heatmap(dataset.corr(), annot=False))
            st.pyplot()
        
        if st.checkbox("Pie Plot"):
            all_columns_names = dataset.columns.tolist()
            fig = px.pie(dataset, values='Album Name', names='Artist Name(s)')
            st.plotly_chart(fig)
        
        if st.checkbox("Show Scatter Plot"):
            # Scatter plot of Danceability vs Energy colored by Popularity
            fig3 = px.scatter(dataset, x='Danceability', y='Energy', color='Popularity', title='Danceability vs Energy by Popularity')
            st.plotly_chart(fig3)

def perform_trend_analysis(dataset):
    # Define the container for the visualization
    visualization_container = st.container()

    # Convert Popularity to numeric data type and duration_ms column to float
    dataset['Duration (ms)'] = dataset['Duration (ms)'].astype(float)
    dataset['Popularity'] = pd.to_numeric(dataset['Popularity'])
    dataset['Release Date'] = pd.to_datetime(dataset['Release Date'])
    # Define start and end dates for trend analysis
    start_date = pd.to_datetime('01-01-2010')
    end_date = pd.to_datetime('31-12-2022')
    
    with visualization_container:
        st.subheader("Trend Analysis")
        
        # Bar plot of Top 10 Artists by Popularity
        top10_artists = dataset.groupby('Artist Name(s)')['Popularity'].mean().sort_values(ascending=False).head(10)
        fig2 = px.bar(x=top10_artists.index, y=top10_artists.values, title='Top 10 Artists by Popularity', labels={'x':'Artist', 'y':'Popularity'})
        st.plotly_chart(fig2)
        
        # Bar plot of Top 10 Genres by Number of Tracks
        top10_genres = dataset.groupby('Genres')['Track Name'].count().sort_values(ascending=False).head(10)
        fig = px.bar(x=top10_genres.index, y=top10_genres.values, title='Top 10 Genres by Number of Tracks', labels={'x':'Genres', 'y':'Number of Tracks'})
        st.plotly_chart(fig)
        
        # Filter the dataset by date range
        filtered_dataset = dataset[(pd.to_datetime(dataset['Release Date']) >= start_date) & (pd.to_datetime(dataset['Release Date']) <= end_date)]

         # Compute average duration by year
        avg_duration = filtered_dataset.groupby(filtered_dataset['Release Date'].dt.year)['Duration (ms)'].mean().reset_index()
        avg_duration['duration_min'] = avg_duration['Duration (ms)'] / 60000
        avg_duration = avg_duration.rename(columns={'Release Date': 'Year'})

        # Visualize trend in average duration
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=avg_duration['Year'], y=avg_duration['duration_min'], mode='lines+markers'))
        fig.update_layout(title='Trend in Average Song Duration',
                          xaxis_title='Year',
                          yaxis_title='Duration (minutes)')
        st.plotly_chart(fig)
        
        # Plot histogram of track durations
        st.subheader("Histogram Plot for Track Duration")
        fig = px.histogram(dataset, x='Duration (ms)', nbins=50)
        st.plotly_chart(fig)
        
        # Line plot of Number of Tracks by Year
        tracks_by_year = dataset.groupby(dataset['Release Date'].dt.year)['Track Name'].count().reset_index()
        tracks_by_year = tracks_by_year.rename(columns={'Track Name': 'Number of Tracks', 'Release Date': 'Year'})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tracks_by_year['Year'], y=tracks_by_year['Number of Tracks'], mode='lines+markers'))
        fig.update_layout(title='Number of Tracks by Year',
                          xaxis_title='Year',
                          yaxis_title='Number of Tracks')
        st.plotly_chart(fig)
        # Trend in Energy
        st.subheader("Trend in Energy")
        energy_trend = dataset.groupby(dataset['Release Date'].dt.year)['Energy'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=energy_trend['Release Date'], y=energy_trend['Energy'], mode='lines+markers'))
        fig.update_layout(title='Trend in Energy',
                          xaxis_title='Year',
                          yaxis_title='Energy')
        st.plotly_chart(fig)
        
       # Pie chart for Valence distribution
        st.subheader("Valence Distribution")
        valence_distribution = dataset['Valence']
        labels = ['Negative', 'Neutral', 'Positive']
        values = [valence_distribution[valence_distribution <= 0.3].count(),
                  valence_distribution[(valence_distribution > 0.3) & (valence_distribution <= 0.7)].count(),
                  valence_distribution[valence_distribution > 0.7].count()]
        colors = ['rgb(255, 0, 0)', 'rgb(255, 255, 0)', 'rgb(0, 128, 0)']

        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
        fig.update_layout(title='Valence Distribution')
        st.plotly_chart(fig)

        # Trend in Tempo
        st.subheader("Trend in Tempo")
        tempo_trend = dataset.groupby(dataset['Release Date'].dt.year)['Tempo'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tempo_trend['Release Date'], y=tempo_trend['Tempo'], mode='lines+markers'))
        fig.update_layout(title='Trend in Tempo',
                          xaxis_title='Year',
                          yaxis_title='Tempo')
        st.plotly_chart(fig)
        
def user_behavior_analysis():
    dataset_container = st.container()

    with dataset_container:
        st.subheader("User Behavior Analysis")
        
        # File upload section
        st.subheader("Upload User Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file, encoding='unicode_escape', engine='python')
            
            # Check top songs
            st.subheader("Most Played Song")
            most_played_song = user_data['Track Name'].value_counts().idxmax()
            st.write("Most Played Song:", most_played_song)
            
            # Check most loved genre
            genre_counts = user_data['Genres'].value_counts()
            most_loved_genre = genre_counts.idxmax()
            st.subheader("Most Loved Genre")
            st.write("Most Loved Genre:", most_loved_genre)
            
            # Number of unique artists
            st.subheader("Unique artists")
            num_unique_artists = user_data['Artist Name(s)'].nunique()
            st.write("Number of Unique Artists:", num_unique_artists)
            
            # Average popularity of songs
            st.subheader("Average Popularity ")
            avg_popularity = user_data['Popularity'].mean()
            st.write("Average Popularity:", avg_popularity)
            
            #  Top artists
            top_artists = user_data['Artist Name(s)'].value_counts().head(5)
            st.subheader("Top Artists")
            st.write(top_artists)
            
            #Track Count
            track_counts = user_data['Track Name'].value_counts().head(10)
            st.subheader("Track Counts")
            st.write(track_counts)
            
            # Number of unique albums
            st.subheader("Unique Albums")
            num_unique_albums = user_data['Album Name'].nunique()
            st.write("Number of Unique Albums:", num_unique_albums)

            # Top 5 albums
            top_albums = user_data.groupby('Album Name').size().sort_values(ascending=False).head(5)
            st.subheader("Top 5 Albums")
            st.write(top_albums)

            # Check most popular songs (excluding duplicate albums)
            st.subheader("Most Popular Songs")
            popular_songs = dataset.drop_duplicates(subset=["Album Name"])[["Track Name", "Artist Name(s)", "Popularity"]]
            popular_songs = popular_songs.sort_values("Popularity", ascending=False).head(10)
            st.table(popular_songs)

            # Bar plot of Top 20 Artists by Popularity
            top20_artists = dataset.groupby('Artist Name(s)')['Popularity'].mean().sort_values(ascending=False).head(20)
            fig2 = px.bar(x=top20_artists.index, y=top20_artists.values, title='Top 20 Artists by Popularity', labels={'x':'Artist', 'y':'Popularity'})
            st.plotly_chart(fig2)

            # Bar plot of Top 20 Genres by Number of Tracks
            top20_genres = dataset.groupby('Genres')['Track Name'].count().sort_values(ascending=False).head(20)
            fig = px.bar(x=top20_genres.index, y=top20_genres.values, title='Top 20 Genres by Number of Tracks', labels={'x':'Genres', 'y':'Number of Tracks'})
            st.plotly_chart(fig)
            # Check added by distribution
            st.subheader("Added By Distribution")
            added_by_counts = dataset['Added By'].value_counts()
            st.bar_chart(added_by_counts)
            
            # Determine favorite song of each Added By user
            st.subheader("Favorite Song by Added By User")
            favorite_songs = dataset.groupby('Added By')['Popularity'].idxmax()
            favorite_songs_data = dataset.loc[favorite_songs, ['Added By', 'Track Name', 'Artist Name(s)', 'Popularity']]

            fig = px.bar(favorite_songs_data, x='Added By', y='Popularity', color='Track Name', title='Favorite Song by Added By User')
            st.plotly_chart(fig)
            # Determine favorite genre of each Added By user
            st.subheader("Favorite Genre by Added By User")
            favorite_genres = dataset.groupby('Added By')['Genres'].agg(lambda x: x.value_counts().index[0])
            favorite_genres_data = favorite_genres.reset_index()

            fig = px.bar(favorite_genres_data, x='Added By', y='Genres', color='Genres', title='Favorite Genre by Added By User')
            st.plotly_chart(fig)


            

        
def main():
        st.set_page_config(page_title="Spotify Data Analysis", page_icon=":notes:")
        # Define the choices
        choices = ['Home', 'EDA', 'Trend Analysis', 'User Behaviour Analysis']

        # Create the sidebar
        choice = st.sidebar.selectbox("Select an option", choices)

        # Main page logic

        if choice == 'Home':
            home_section(dataset)
        elif choice == 'EDA':
            perform_eda(dataset)
        elif choice == 'Trend Analysis':
            perform_trend_analysis(dataset)
        elif choice == 'User Behaviour Analysis':
            user_behavior_analysis()

if __name__ == "__main__":
    main()
    








