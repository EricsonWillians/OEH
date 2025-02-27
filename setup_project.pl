#!/usr/bin/perl
use strict;
use warnings;
use File::Path qw(make_path);
use Term::ANSIColor;

# Define directories (Ensure they exist before creating files)
my @dirs = qw(
    oeh
    oeh/simulation
    oeh/rendering
    oeh/rendering/shaders
    oeh/cuda
    utils
    tests
);

# Define files (Each file must have a valid parent directory)
my @files = qw(
    uv.toml
    requirements.txt
    oeh/__init__.py
    oeh/main.py
    oeh/config.py
    oeh/simulation/__init__.py
    oeh/simulation/integrator.py
    oeh/simulation/raytracer.py
    oeh/rendering/__init__.py
    oeh/rendering/opengl_renderer.py
    oeh/rendering/shaders/vertex_shader.glsl
    oeh/rendering/shaders/fragment_shader.glsl
    oeh/cuda/__init__.py
    oeh/cuda/kernel.py
    utils/__init__.py
    utils/logger.py
    utils/helpers.py
    tests/__init__.py
    tests/test_simulation.py
    tests/test_rendering.py
);

# Create directories
foreach my $dir (@dirs) {
    if (! -d $dir) {
        make_path($dir) or die "ERROR: Could not create directory $dir: $!\n";
        print color("green"), "Created directory: $dir\n", color("reset");
    } else {
        print color("yellow"), "Directory already exists: $dir\n", color("reset");
    }
}

# Create files
foreach my $file (@files) {
    # Extract parent directory from file path (if any)
    if ($file =~ m|/|) {
        my ($parent_dir) = $file =~ m|^(.*)/[^/]+$|;
        make_path($parent_dir) if (! -d $parent_dir);  # Ensure directory exists
    }

    if (! -e $file) {
        open(my $fh, '>', $file) or die "ERROR: Could not create file $file: $!\n";
        close $fh;
        print color("green"), "Created file: $file\n", color("reset");
    } else {
        print color("yellow"), "File already exists: $file\n", color("reset");
    }
}

print color("cyan"), "✅ Project structure initialized successfully.\n", color("reset");
