%% augment001 %%
clear all
%% SETUP
OUTdir='/esat/spchtemp/scratch/jponcele/cgn_vl_telephone_sequences_augmented/';
add_filter=false;
add_noise=true;
min_SNR=5;  %0 or 5
max_SNR=20;  %15 or 20
%%
% CGN files
%CGNdir='/users/spraak/spchdata/cgn/wav/';
%CGNfiles=dir([CGNdir 'comp-*/vl/*.wav']);
CGNdir='/esat/spchtemp/scratch/jponcele/cgn_vl_telephone_sequences/wav/';
CGNfiles=dir([CGNdir 'comp-*/vl/*.wav']);

% sampling rate 48kHz
RIRdir='/esat/spchdisk/scratch/hvanhamm/ESPnet/cgn/AIR_1_4/';
RIRfiles=dir([RIRdir '*.mat']);

% sampling rate 22kHz, each file 3 minutes
NTTdir='/users/spraak/spchdata/NTTambient/CDdata/';
NTTfiles1=dir([NTTdir 'vol1_*/*.22k']); % many out-dated cars
NTTfiles2=dir([NTTdir 'vol2_*/*.22k']);

% sampling rate 20kHz
NOISEXdir='/users/spraak/spchdata/noisex/CDdata_noise_rom_0/data/';
NOISEXfiles=dir([NOISEXdir 'signal.*']);

% sampling rate 16kHz, 5 minutes each
DEMANDdir='/esat/spchdisk/scratch/hvanhamm/ESPnet/cgn/demand/';
DEMANDfiles=dir([DEMANDdir '*.wav']);

% sampling rate 16kHz, most 5 minutes
CHIMEdir='/users/spraak/spchdata/chime2/PCCdata16kHz/train/background/';
CHIMEfiles=dir([CHIMEdir '*.wav']);

% sampling rate 16kHz, 5 mins
HUMMdir='/esat/spchdisk/scratch/hvanhamm/ESPnet/cgn/humm/';
HUMMfiles=dir([HUMMdir '*.wav']);

ALLfiles =  {NTTfiles1,NTTfiles2,NOISEXfiles, DEMANDfiles,CHIMEfiles,HUMMfiles};
nsource = {'NTTambient_vol1','NTTambient_vol2','NOISEX','DEMAND','CHIME','humm'};
Fraction = [1  5   2    5    5   7]; % draw from noise sources proportional to these numbers

cumFraction=l_expand(1:length(Fraction),Fraction); % needed for sampling
[downB,downA] = butter(4,1/3); % butterworth filter for converting 48kHz to 16kHz

addpath('/users/spraak/hvanhamm/matlab/sigproc','-END');
%addpath(genpath('/users/spraak/hvanhamm/matlab'))

%%
if add_noise & add_filter
    suffix='_noisyandfiltered';
    outfile = fullfile(OUTdir,strcat('summary',suffix,'.txt'));
    fileID = fopen(outfile, 'w');
    fprintf(fileID,'% fileid SNR noisetype noisesrc - RIRtype RIRlength\n')
elseif add_noise
    suffix='_noisy1';
    outfile = fullfile(OUTdir,strcat('summary',suffix,'.txt'));
    fileID = fopen(outfile, 'w');
    fprintf(fileID,'% fileid SNR noisetype noisesrc\n')
elseif add_filter
    suffix='_filtered';
    outfile = fullfile(OUTdir,strcat('summary',suffix,'.txt'));
    fileID = fopen(outfile, 'w');
    fprintf(fileID,'% fileid RIRtype RIRlength\n')
else
    error('Choose add_filter, add_noise, or both');
end


%%
%wb = waitbar(0,'Augmenting data...')
for k=1:length(CGNfiles)
    %msg=sprintf('Augmenting data: %3.1f%% completed', k/length(CGNfiles))
    %waitbar(k/length(CGNfiles),wb,msg)
    [sam,fs]=audioread(fullfile(CGNfiles(k).folder,CGNfiles(k).name));
    fprintf('%s : %8.3fs - %d Hz\n',CGNfiles(k).name,length(sam)/fs,fs);
    dur=length(sam)/fs;
    if fs<16000,continue;end
    sam=resample(sam,16000,fs);
    dBsig=quantile(10*(logeonly(filter([1 -0.98],1,sam),480,160,1e-8)/log(10)),0.9);
    augname=insertBefore(CGNfiles(k).name,'.wav',suffix);
    %fprintf(fileID,'---------------------------------------------------\n');
    %fprintf(fileID, '%s : %8.3f s - %d Hz - %i sam\n', insertBefore(CGNfiles(k).name,'.wav',suffix), dur, fs, length(sam));
    
    % filter
    if add_filter
        while true
            i=randi(length(RIRfiles));
            load(fullfile(RIRfiles(i).folder,RIRfiles(i).name));
            h_air=filtfilt(downB,downA,h_air(:));h_air=h_air(1:3:end);
            % keep only most important part
            x=cumsum(flipud(h_air).^2);TooSmall=flipud(x/x(end)<1e-5);h_air(TooSmall)=[];
            % resample probabilistically, depending on length
            %fprintf('   %s (%d)',air_info.room,length(h_air));
            switch air_info.room
                case 'aula_carolina'
                    if rand(1)<0.02,break;end
                case 'stairway'
                    if rand(1)<0.05,break;end
                case 'lecture'
                    if rand(1)<0.5,break;end
                case 'meeting'
                    if rand(1)<0.7,break;end
                otherwise
                    break;
            end
            %if length(h_air)/120000<rand(1),break;end
        end
        sam=filter(h_air,1,[zeros(length(h_air)-1,1);mean(sam,2)]);
        sam=sam(length(h_air):end);
        dBsig2=quantile(10*(logeonly(filter([1 -0.98],1,sam),480,160,1e-8)/log(10)),0.9);
        sam=10^((dBsig-dBsig2)/20)*sam;
    end
    
    %add noise
    if add_noise
        noise=[];
        while length(noise)<length(sam)
            iSource=cumFraction(randi(length(cumFraction)));
            iFile=randi(length(ALLfiles{iSource}));
            switch iSource
                case {1,2}, % NTT
                    fid=fopen(fullfile(ALLfiles{iSource}(iFile).folder,ALLfiles{iSource}(iFile).name),'rb','l');
                    NoiseSam=resample(fread(fid,[1 inf],'int16')'/(2^15),16000,22000);
                    fclose(fid);
                case 3, % NOISEX
                    fid=fopen(fullfile(ALLfiles{iSource}(iFile).folder,ALLfiles{iSource}(iFile).name),'rb','l');
                    NoiseSam=resample(fread(fid,[1 inf],'int16')'/(2^15),16000,20000);
                    fclose(fid);
                otherwise,
                    [NoiseSam,fs]=audioread(fullfile(ALLfiles{iSource}(iFile).folder,ALLfiles{iSource}(iFile).name));
                    NoiseSam=resample(NoiseSam,16000,fs);
            end
            NoiseSam=filter([1 -1],1,NoiseSam);
            % Take random 1 minute (max) sample
            iStart=randi(max(length(NoiseSam)-16000*60,0)+1);
            NoiseSam=NoiseSam(iStart:min(length(NoiseSam),iStart+16000*60-1));
            dBnoise=quantile(10*(logeonly(NoiseSam,480,160,1e-8)/log(10)),0.9);
            %SNR=5+15*rand(1); %gv-files
            %SNR=15*rand(1); % hv files
            SNR=min_SNR+(max_SNR-min_SNR)*rand(1);
            Scale=10^((dBsig-dBnoise-SNR)/20);
            start=length(noise);
            noise=[Scale*NoiseSam(:)];
            endpoint=min(length(noise), length(sam));
            noisetype=nsource{iSource};
            noisename=ALLfiles{iSource}(iFile).name;
            %fprintf(fileID,'%i %i -  %3.1f SNR - %s %s\n',start,endpoint,SNR,noisetype,noisename);
        end
        sam=sam+noise(1:length(sam));
    end
    
    if add_noise & add_filter
        fprintf(fileID,'%s %3.1f %s %s - %s %d\n',augname,SNR,noisetype,noisename,air_info.room,length(h_air));
    elseif add_noise
        fprintf(fileID,'%s %3.1f %s %s\n',augname,SNR,noisetype,noisename);
    else
        fprintf(fileID,'%s %s %d\n',augname,air_info.room,length(h_air));
    end
    slpos=find(CGNfiles(k).folder == filesep);
    fDir=[OUTdir CGNfiles(k).folder(slpos(end-1)+1:end)];
    if ~exist(fDir,'dir')
        mkdir(fDir);
    end
    audiowrite([fDir filesep insertBefore(CGNfiles(k).name,'.wav',suffix)],sam,16000); 
    %audiowrite([fDir filesep strrep(CGNfiles(k).name,'fv','iv')],sam,16000);
    %fprintf('%s : %8.3fs - %d Hz\n',CGNfiles(k).name,length(sam)/fs,fs);
end

fclose(fileID)
%waitbar(1,wb,'Finished')
%pause(1)
%close(wb)
fprintf('FINISHED')
