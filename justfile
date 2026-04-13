linux-build-nuget:
    cd ~/src/diskann
    cargo build --release --package diskann-garnet
    cp diskann-garnet/diskann-garnet.nuspec ~/tmp/pkg/
    cp target/release/libdiskann_garnet.so ~/tmp/pkg/runtimes/linux-x64/native/
    rm -f ~/tmp/diskann-garnet.1.*.nupkg
    cd ~/tmp/pkg ; zip -r /home/jackmoffitt/tmp/diskann-garnet.1.0.26.nupkg *
    sudo cp ~/tmp/diskann-garnet.1.0.26.nupkg /usr/lib/dotnet/library-packs/
    dotnet nuget locals all --clear

linux-build-nuget-debug:
    cd ~/src/diskann
    cargo build --package diskann-garnet
    cp diskann-garnet/diskann-garnet.nuspec ~/tmp/pkg/
    cp target/debug/libdiskann_garnet.so ~/tmp/pkg/runtimes/linux-x64/native/
    rm -f ~/tmp/diskann-garnet.1.*.nupkg
    cd ~/tmp/pkg ; zip -r /home/jackmoffitt/tmp/diskann-garnet.1.0.26.nupkg *
    sudo cp ~/tmp/diskann-garnet.1.0.26.nupkg /usr/lib/dotnet/library-packs/
    dotnet nuget locals all --clear
