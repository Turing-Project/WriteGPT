  
#!/usr/bin/env perl
#
# Google Drive direct download of big files
# ./gdown.pl 'gdrive file url' ['desired file name']
#
# v1.0 by circulosmeos 04-2014.
# v1.1 by circulosmeos 01-2017.
# v1.2, 2.0 by circulosmeos 01-2019.
# //circulosmeos.wordpress.com/2014/04/12/google-drive-direct-download-of-big-files
# Distributed under GPL 3 (//www.gnu.org/licenses/gpl-3.0.html)
#
use strict;
use POSIX;

my $TEMP='gdown.cookie.temp';
my $COMMAND;
my $confirm;
my $check;
sub execute_command();

my $URL=shift;
die "\n./gdown.pl 'gdrive file url' [desired file name]\n\n" if $URL eq '';

my $FILENAME=shift;
my $TEMP_FILENAME='trained_models/model.ckpt-280000.data-00000-of-00001';

if ($URL=~m#^https?://drive.google.com/file/d/([^/]+)#) {
    $URL="https://docs.google.com/uc?id=$1&export=download&confirm=t";
}
elsif ($URL=~m#^https?://drive.google.com/open\?id=([^/]+)#) {
    $URL="https://docs.google.com/uc?id=$1&export=download&confirm=t";
}

execute_command();

while (-s $TEMP_FILENAME < 100000) { # only if the file isn't the download yet
    open fFILENAME, '<', $TEMP_FILENAME;
    $check=0;
    foreach (<fFILENAME>) {
        if (/href="(\/uc\?export=download[^"]+)/) {
            $URL='https://docs.google.com'.$1;
            $URL=~s/&amp;/&/g;
            $confirm='';
            $check=1;
            last;
        }
        if (/confirm=([^;&]+)/) {
            $confirm=$1;
            $check=1;
            last;
        }
        if (/"downloadUrl":"([^"]+)/) {
            $URL=$1;
            $URL=~s/\\u003d/=/g;
            $URL=~s/\\u0026/&/g;
            $confirm='';
            $check=1;
            last;
        }
    }
    close fFILENAME;
    die "Couldn't download the file :-(\n" if ($check==0);
    $URL=~s/confirm=([^;&]+)/confirm=$confirm/ if $confirm ne '';

    execute_command();

}

unlink $TEMP;

sub execute_command() {
    my $OUTPUT_FILENAME = $TEMP_FILENAME;
    my $CONTINUE = '';

    # check contents before download & if a $FILENAME has been indicated resume on content download
    # please, note that for this to work, wget must correctly provide --spider with --server-response (-S)
    if ( length($FILENAME) > 0 ) {
        $COMMAND="wget -q -S --no-check-certificate --spider --load-cookie $TEMP --save-cookie $TEMP \"$URL\" 2>&1";
        my @HEADERS=`$COMMAND`;
        foreach my $header (@HEADERS) {
            if ( $header =~ /Content-Type: (.+)/ ) {
                if ( $1 !~ 'text/html' ) {
                    $OUTPUT_FILENAME = $FILENAME;
                    $CONTINUE = '-c';
                }
            }
        }
    }

    $COMMAND="wget $CONTINUE --progress=dot:giga --no-check-certificate --load-cookie $TEMP --save-cookie $TEMP \"$URL\"";
    $COMMAND.=" -O \"$OUTPUT_FILENAME\"";
    my $OUTPUT = system( $COMMAND );
    if ( $OUTPUT == 2 ) { # do a clean exit with Ctrl+C
        unlink $TEMP;
        die "\nDownloading interrupted by user\n\n";
    } elsif ( $OUTPUT == 0 && length($CONTINUE)>0 ) { # do a clean exit with $FILENAME provided
        unlink $TEMP;
        die "\nDownloading complete\n\n";
    }
    return 1;
}
