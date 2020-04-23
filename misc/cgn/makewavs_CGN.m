% makewavs003: write phone labels in talab-file and wavs
% CGN-data orthographic transcriptions
% take all speech of the same speaker in a file together.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%SETUP%%%
indir='/users/spraak/spchdata/cgn/CDdata/CGN_V2.0_tst_ann/data/annot/text/awd/';
cgndir='/users/spraak/spchdata/cgn/wav/';

%outdir='/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_afgklno/wav/';
%talabfile='/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_afgklno/fac/all_facs_phones.scp';
%components='afgklno';

% make an empty 'wav' and 'fac' directory first
outdir='/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_ko/wav/';
talabfile='/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/cgn_np_fbank_ko/fac/all_facs_phones.scp';
components='ko';


pho={'[]','sil','p','t','k','b','d','g','f','v','s','z','S','Z','x','G',...
    'h','N','m','n','J','l','r','w','j','I','E','A','O','Y','i','y','e',...
    '2','a','o','u','@','E+','Y+','A+','E:','Y:','O:','E~','A~','O~','#'};

TargetLength = 2000; % target length of cut wav files, in frames of 10ms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tafid=fopen(talabfile,'wb');
for comp=1:length(components)
seldir=['comp-' components(comp) '/vl/'];
unix(['mkdir -p ' outdir seldir]);
if ismember(components(comp),{'f','l'}) suff='_A0'; else suff=''; end;

files=dir([indir seldir '*.awd.gz']);
for k=1:length(files)
  fprintf('\n%s %d/%d: %s - ',components(comp),k,length(files),files(k).name);
  gunzip(fullfile(files(k).folder,files(k).name),'.');
  fileid=files(k).name(1:end-7);
  if ~exist([cgndir seldir fileid suff '.wav'],'file') continue;end
  [sam,fs]=audioread([cgndir seldir fileid suff '.wav']);
  if fs~=16000 error('sample freq'); end
  sam=mean(sam,2);
  Nseg=0;Nentry=0;
  fid=fopen(files(k).name(1:end-3));
  C=textscan(fid,'%s',1,'delimiter','\n','headerlines',6);C=C{1}; 
  nTiers=str2num(C{1});
  nSpk=0;
  filelen=0;
  clear spkid trans starttimes endtimes
  for tier=1:nTiers
    D=textscan(fid,'%s',5,'delimiter','\n');D=D{1};
    nFields=str2num(D{5});
    C=textscan(fid,'%s',3*nFields,'delimiter','\n');C=C{1};
    if ~any(ismember(D{2},'0123456789')) | ~strcmp(D{2}(end-4:end-1),'_SEG') %check if a speaker number is in the description field
        continue;
    end 
    nSpk=nSpk+1;
    spkid{nSpk}=lower(D{2}(1:end-5));spkid{nSpk}(spkid{nSpk}=='"')=[];
    instance=0;
    fprintf(' %s',spkid{nSpk})
    Nentry=Nentry+1;
    starttimes{nSpk}=cellfun(@str2num,C(1:3:end));
    endtimes{nSpk}=cellfun(@str2num,C(2:3:end));
    if endtimes{nSpk}(end)>filelen filelen=endtimes{nSpk}(end);end
    strs=strrep(C(3:3:end),'"','');
    strs=strrep(strs,'!',''); % ! indicates unreliable label. Ignore.
    strs=strrep(strs,'_',''); % _ indicates shared phone over words. Ignore.
    strs=strrep(strs,'-',''); % - indicates linking phone. Ignore.
    strs=strtrim(strs);
    sel=cellfun(@length,strs)==0;
    %strs(sel)={'sil'};
    % remove silence: it belongs to no-one
    starttimes{nSpk}(sel)=[];
    endtimes{nSpk}(sel)=[];
    strs(sel)=[];
    trans{nSpk}=strs;
    [~,lab{nSpk}]=ismember(strs,pho);
    lab{nSpk}=clip0(lab{nSpk}-1); % # 0 signifies [] or unknown    
  end  
  fclose(fid);
  
  spknr=zeros(1,ceil(filelen*100)); % which speaker per 10ms segment
                % if multiple speakers active, sum of the speaker ids
                % hence if spkid > nSpk => multiple active
  label=ones(1,ceil(filelen*100)); % silence by default
  for spk=1:nSpk,
    stst=ceil(100*[starttimes{spk}';endtimes{spk}'])+[1;0];
    len=num2cell(stst(2,:)-stst(1,:)+1);
    o=num2cell(ones(1,length(len)));
    z=cellfun(@repmat,num2cell(lab{spk}'),o,len,'uniformoutput',false);z=[z{:}];
    sel=dbl_pnt(stst);
    % never overwrite with silence (label 1)
    cnd=(z==1)&(label(sel)~=1);
    sel(cnd)=[];z(cnd)=[];
    spknr(sel)=spknr(sel)+spk+nSpk;
    label(sel)=z;
  end
  spknr=clip0(spknr-nSpk);
  
  % silence gaps assigned to noone
  [it,frq]=packul((spknr==0)&(label==1));
  segend=cumsum(frq);
  segbeg=segend-frq+1;
  spknr2=[0 spknr 0];
  leftspk=spknr2(segbeg);
  rightspk=spknr2(segend+2);
  % silence segments without assigned speaker of length less than 50
  sel=(leftspk==rightspk)&(frq<50)&(leftspk<=nSpk)&(it==1);
  for k=find(sel),
    spknr(segbeg(k):segend(k))=leftspk(k);
  end
  alarm = length(sam)/fs*100 < length(label)-1;
  spknr(floor(length(sam)/fs*100):end)=0;
  
  if alarm fprintf('\nCheck:');end
  for spk=1:nSpk,  
    instance=0;
    wavfname=[outdir seldir spkid{spk} '_%d.wav'];
    if alarm fprintf(['\n  ' spkid{spk} ' : ']);end
    rng=find(spknr==spk);
    [it,frq]=packul(label(rng));
    endpoints=cumsum(frq);
    if isempty(endpoints) continue;end
    nSeg=max(1,ceil(endpoints(end)/TargetLength));
    cutpoints=(1:nSeg-1)/nSeg*endpoints(end);
    % find endpoint closest to cutpoints
    ii=find(it==1); % only cut on label = 1 (silence)
    dx=bsxfun(@minus,endpoints(ii)',cutpoints).^2;
    [~,argmin]=min(dx,[],1);
    cutpoints=[0 endpoints(ii(argmin)) endpoints(end)];
    jj=1;
    instance=0;
    for kkk=2:length(cutpoints)
        frames=rng(cutpoints(kkk-1)+1:cutpoints(kkk));
        thisspeaker=sam(dbl_pnt([fs/100*frames+1;fs/100*(frames+1)]));
        while exist(sprintf(wavfname,instance),'file') instance=instance+1;end
        audiowrite(sprintf(wavfname,instance),thisspeaker,fs);
        if alarm fprintf('%d ',instance);end
        fprintf(tafid,[spkid{spk} '_%d_%s\n'],instance,components(comp));
        t0=endpoints(jj)-frq(jj);
        while (jj<=length(endpoints)) & (endpoints(jj)<=cutpoints(kkk))
            fprintf(tafid,'%d %d %d\n',endpoints(jj)-frq(jj)-t0,endpoints(jj)-t0,it(jj));
            jj=jj+1;
        end
        instance=instance+1;    end
  end
  delete *.awd
end

end % comp
fclose(tafid);

function [Items,Freq]=packul(List);
% packul:   return Freq(uency) with which Items occur in ORDERED List.
% [Items,Freq]=packul(List);
% e.g: [Items,Freq]=packul([4 4 4 5 5 6 4 4])
%      yields: Items=[4 5 6 4]
%              Freq =[3 2 1 2]
if isempty(List),
  Items=[];
  Freq=[];
  return;
end
List=List(:)';
Flag=~(~[1 fdiff(List)]);
Items=List(Flag);
T=length(List);
ct=1:T;
Freq=fdiff([ct(Flag) T+1]);
end

function y=dbl_pnt(x)
% dbl_pnt: generalise the ":" symbol
% x=[Start;Stop]
% y=[Start(1,1):Stop(2,1) Start(1,2):Stop(2,2) ... Start(1,N):Stop(2,N)]
c=cellfun(@colon,num2cell(x(1,:)),num2cell(x(2,:)),'uniformoutput',false);
y=[c{:}];
end

function y=clip0(x)
% clip0:    clip values below 0 in a matrix.
% y=clip0(x)
% y=x.*(x>0);
y=x;
if isempty(x),
  return
end
sel=(x<=0);
y(sel)=zeros(sum(sum(sel)),1);
end

function y=fdiff(x)
% fdiff:    y=(1-z^-1)*x
T=size(x,2);
y=x(:,2:T)-x(:,1:T-1);
end