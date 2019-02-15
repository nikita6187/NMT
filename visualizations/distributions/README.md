# Visualizations of distributions

## Installation

- Note: Max batch size of 1 is allowed!

- Insert the following code into TFNetworkLayer.py in lines 5679, just before the `return self.reduce_func(out)`:
`out = tf.Print(out, [self.output.get_placeholder_as_time_major()], message="Softmax:", summarize=10000)
out = tf.Print(out, [self.target.get_placeholder_as_time_major()], message="Target:", summarize=10000)`

- Run the visualiztion.py script
- TODO: explain how it works