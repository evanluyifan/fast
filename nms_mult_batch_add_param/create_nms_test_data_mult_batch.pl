#!/usr/bin/perl
#
# user input
my $batch = 12;
# user input

my $box_num_per_batch = 6000;
my $box_num = $box_num_per_batch * $batch;

my $fo = "testdata.h";
open(my $fout, ">", $fo) || die "Couldn't open '".$fo."' for reading because: ".$!;

my $box_size = 4;
my $array_size = $box_num * $box_size;

print $fout "#define BATCH ".$batch."\n";
print $fout "#define TEST_BOXES_NUM_PER_BATCH ".$box_num_per_batch."\n";
print $fout "#define ARRAY_SIZE ".$array_size."\n";

print $fout "float test_boxes[ARRAY_SIZE] = {\n";

my $count = 0;

my $x_0 = 1.5;
my $y_0 = 1.5;
my $x_1 = 2.5;
my $y_1 = 2.5;

my $end_loop = $box_num / 4;

while ($count < $end_loop) {
    print $fout $x_0.", ".$y_0.", ".$x_1.", ".$y_1.",\n";
    print $fout $x_0.", ".$y_0.", ".$x_1.", ".$y_1.",\n";
    print $fout $x_0.", ".$y_0.", ".$x_1.", ".$y_1.",\n";
    print $fout $x_0.", ".$y_0.", ".$x_1.", ".$y_1.",\n";
    $x_0 += 1.0;
    $y_0 += 1.0;
    $x_1 += 1.0;
    $y_1 += 1.0;
    $count = $count + 1;
}

print $fout "};\n";

print "Output file - ".$fo." created\n";
