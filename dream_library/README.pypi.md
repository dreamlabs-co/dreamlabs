<!-- The PyPI page of dreamlabs (pyproject.toml, `readme`): the public repository's root README.md with the logo at an
absolute URL, since PyPI resolves no relative path and drops <source>. Change the two together. -->

<div align="center">

<img src="https://raw.githubusercontent.com/dreamlabs-co/dreamlabs/main/assets/dreamlabs-logo-light.png" alt="Dream Labs" width="520">

<p>
  <a href="https://dreamlabs.ai"><strong>Blog Post</strong></a> |
  <a href="https://platform.dreamlabs.ai"><strong>DreamAPI</strong></a> |
  <a href="https://platform.dreamlabs.ai/api-documentation"><strong>Documentation</strong></a>
</p>

</div>

## What is this library for

`dreamlabs` is the Python client for DreamAPI, which hosts **Dream-1**, the Dream Labs robot model. Dream-1 looks at
your robot's cameras and joint positions, reads the task you type, and answers with the next joint targets for the
arms, zero-shot, with no training on your robot.

It comes two ways:

- **`dreamcli`** runs your robot on our control loop: setup, cameras, arms and Live View, with no code to write.
- **`dream.DreamClient`** calls Dream-1 from your own control loop: you send observations, it returns actions.

## Quick Guide

Sign in at [platform.dreamlabs.ai](https://platform.dreamlabs.ai/api-keys) and create an API key. It is shown only
once, so keep it somewhere safe.

### dreamcli: our control loop

```bash
pip install dreamlabs
dreamcli setup       # arms and cameras
dreamcli robot up    # start the robot
```

Give it a task from [Live View](https://platform.dreamlabs.ai/live). For the full walkthrough, open
[Robot Setup](https://platform.dreamlabs.ai/setup).

### DreamClient: your own control loop

```bash
pip install dreamlabs-client
export DREAM_API_KEY=dl_live_...
```

```python
import dream

client = dream.DreamClient()
```

To run a model from your loop, see the [documentation](https://platform.dreamlabs.ai/api-documentation).

## Documentation

The [DreamAPI documentation](https://platform.dreamlabs.ai/api-documentation) covers everything beyond the quick
guide: getting started, calling Dream-1 from your own control loop with `dream.DreamClient`, showing your robot in
Live View, bringing a robot of your own, and fine-tuning Dream-1 on recordings of your robot.

## License

Apache License 2.0 (`LICENSE`). Three of the vendored packages, `i2rt`, `pyroki` and `jaxls` under
`dream_library/vendor/`, keep their own MIT licenses, and the robot models and example assets that came from elsewhere
keep theirs, each in a `LICENSE` file in its own folder.
