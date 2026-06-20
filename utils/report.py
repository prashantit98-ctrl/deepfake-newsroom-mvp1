def generate_report(metadata, transcript, frame_count):
    risk='LOW'
    if any(w in transcript.lower() for w in ['exclusive','breaking','leaked']):
        risk='MEDIUM'
    return {
      'status':'completed',
      'risk_level':risk,
      'frames_analyzed':frame_count,
      'transcript':transcript[:1000],
      'metadata':metadata[:1000]
    }
