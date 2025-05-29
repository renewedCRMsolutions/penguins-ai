# api/game_tracker.py
async def track_live_game(game_id: str):
    pbp = await fetch_game_data(game_id)

    for play in pbp["plays"]:
        if play["typeDescKey"] in ["shot-on-goal", "missed-shot", "goal"]:
            # Calculate real-time features
            time_since_faceoff = calculate_from_pbp(play, pbp)
            shooter_stats = await get_player_stats(play["shooterId"])

            # Make prediction with actual data
            prediction = model.predict(features)
