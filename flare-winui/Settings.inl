// SPDX-License-Identifier: Apache-2.0
    // Backend strings retain their exact units and durations until that field is edited.
    std::unordered_map<std::wstring,std::wstring> runtime_original_,runtime_field_baseline_;
    std::wstring runtime_prefix_directory_;
    double runtime_memory_gib_{2},runtime_memory_percent_{80};
    int runtime_memory_mode_{};
    static std::wstring runtime_decimal(double value) {
        std::wostringstream output;output.imbue(std::locale::classic());
        output<<std::setprecision(15)<<value;return output.str();
    }
    static double runtime_size_gib(std::wstring value) {
        try {
            size_t used{};double amount=std::stod(value,&used);
            std::wstring suffix=value.substr(used);
            suffix.erase(std::remove_if(suffix.begin(),suffix.end(),[](wchar_t c){return std::iswspace(c)!=0;}),suffix.end());
            std::transform(suffix.begin(),suffix.end(),suffix.begin(),[](wchar_t c){return static_cast<wchar_t>(std::towupper(c));});
            double scale=1;
            if(suffix==L"K"||suffix==L"KB"||suffix==L"KIB")scale=1024;
            else if(suffix==L"M"||suffix==L"MB"||suffix==L"MIB")scale=1048576;
            else if(suffix==L"G"||suffix==L"GB"||suffix==L"GIB")scale=1073741824;
            else if(suffix==L"T"||suffix==L"TB"||suffix==L"TIB")scale=1099511627776;
            return std::isfinite(amount)?std::abs(amount)*scale/1073741824.0:0;
        }catch(...){return 0;}
    }
    void runtime_slider(wchar_t const* name,double value,double minimum,double maximum,double step) {
        auto slider=control<Slider>(name);
        slider.Maximum(std::max(maximum,value));slider.Minimum(std::min(minimum,value));
        slider.StepFrequency(step);slider.Value(value);
    }
    void runtime_memory_slider(int mode,double value) {
        runtime_slider(L"RuntimeMemoryValue",value,mode==3?1:mode==1?0.125:0,mode==3?100:128,mode==3?1:0.125);
        runtime_memory_mode_=mode;
    }
    void runtime_memory_mode_changed() {
        if(runtime_settings_applying_)return;
        double previous=control<Slider>(L"RuntimeMemoryValue").Value();
        if(runtime_memory_mode_==3)runtime_memory_percent_=previous;else if(runtime_memory_mode_!=0)runtime_memory_gib_=previous;
        int mode=control<ComboBox>(L"RuntimeMemoryMode").SelectedIndex();
        runtime_settings_applying_=true;
        runtime_memory_slider(mode,mode==3?runtime_memory_percent_:std::max(mode==1?0.125:0.0,runtime_memory_gib_));
        runtime_settings_applying_=false;runtime_settings_changed();
    }
    void runtime_duration(wchar_t const* name,std::wstring const& value,std::vector<std::wstring> const& presets) {
        auto choice=control<ComboBox>(name);
        while(choice.Items().Size()>presets.size())choice.Items().RemoveAtEnd();
        for(size_t i=0;i<presets.size();++i)if(value==presets[i]||(i==0&&value.empty())){choice.SelectedIndex(static_cast<int>(i));return;}
        choice.Items().Append(box_value(hstring(L"Custom: "+value)));choice.SelectedIndex(static_cast<int>(presets.size()));
    }
    std::wstring runtime_field_signature(std::wstring const& key) {
        std::wstring result;
        auto append=[&](std::wstring const& v){result+=std::to_wstring(v.size())+L":"+v;};
        auto choice=[&](wchar_t const* name){append(std::to_wstring(control<ComboBox>(name).SelectedIndex()));};
        auto slider=[&](wchar_t const* name){append(runtime_decimal(control<Slider>(name).Value()));};
        if(key==L"memory"){choice(L"RuntimeMemoryMode");slider(L"RuntimeMemoryValue");}
        else if(key==L"host"){choice(L"RuntimeHostMode");append(std::wstring(control<TextBox>(L"RuntimeHost").Text()));}
        else if(key==L"port")append(runtime_decimal(control<NumberBox>(L"RuntimePort").Value()));
        else if(key==L"kv")choice(L"RuntimeKvLimit");
        else if(key==L"concurrency"){append(std::to_wstring(control<ToggleSwitch>(L"RuntimeConcurrencyAuto").IsOn()));append(runtime_decimal(control<NumberBox>(L"RuntimeConcurrency").Value()));}
        else if(key==L"verify")append(std::to_wstring(control<ToggleSwitch>(L"RuntimeVerify").IsOn()));
        else if(key==L"disk_enabled")append(std::to_wstring(control<ToggleSwitch>(L"RuntimePrefixDiskEnabled").IsOn()));
        else if(key==L"directory")append(runtime_prefix_directory_);
        else if(key==L"ram"){choice(L"RuntimePrefixMemoryMode");slider(L"RuntimePrefixMemoryValue");}
        else if(key==L"disk_limit"){choice(L"RuntimePrefixDiskLimitMode");slider(L"RuntimePrefixDiskLimitValue");}
        else if(key==L"age")choice(L"RuntimePrefixMaxAge");
        else if(key==L"drain")choice(L"RuntimePrefixDrain");
        else if(key==L"storage")choice(L"RuntimeEmbeddingStorage");
        else if(key==L"embedding"){choice(L"RuntimeEmbeddingCacheMode");slider(L"RuntimeEmbeddingCacheValue");}
        else if(key==L"drafts"){choice(L"RuntimeDraftMode");slider(L"RuntimeDraftTokens");}
        else if(key==L"history")choice(L"RuntimeHistoryDrafts");
        else if(key==L"adaptive")choice(L"RuntimeAdaptiveVerification");
        return result;
    }
    std::wstring runtime_form_signature() {
        std::wstring result;
        for(auto key:{L"memory",L"host",L"port",L"kv",L"concurrency",L"verify",L"disk_enabled",L"directory",L"ram",L"disk_limit",L"age",L"drain",L"storage",L"embedding",L"drafts",L"history",L"adaptive"}){
            auto value=runtime_field_signature(key);result+=std::to_wstring(value.size())+L":"+value;
        }
        return result;
    }
    std::wstring runtime_preserve(wchar_t const* key,std::wstring edited) {
        auto found=runtime_field_baseline_.find(key);
        return found!=runtime_field_baseline_.end()&&found->second==runtime_field_signature(key)?runtime_original_[key]:edited;
    }
    std::wstring runtime_budget_value(wchar_t const* mode,wchar_t const* slider) {
        int selected=control<ComboBox>(mode).SelectedIndex();
        return selected==0?L"":selected==1?L"0":runtime_decimal(control<Slider>(slider).Value())+L"GiB";
    }
    void initialize_runtime_settings() {
        clicked(L"RuntimeSettingsSave",[this]{save_runtime_settings();});
        clicked(L"RuntimeSettingsReload",[this]{for(auto const& model:models_)if(!model.catalog()&&(model.flags&FLARE_MODEL_ACTIVE)){activate(model.id,model.name,true);break;}});
        clicked(L"RuntimeSettingsRevert",[this]{runtime_settings_dirty_=false;runtime_settings_initialized_=false;runtime_settings_error_.clear();poll();});
        clicked(L"RuntimePrefixBrowse",[this]{choose_prefix_directory();});
        control<TextBox>(L"RuntimeHost").TextChanged([this](auto const&,auto const&){runtime_settings_changed();});
        control<ComboBox>(L"RuntimeMemoryMode").SelectionChanged([this](auto const&,auto const&){runtime_memory_mode_changed();});
        for(auto name:{L"RuntimeHostMode",L"RuntimeKvLimit",L"RuntimePrefixMemoryMode",L"RuntimePrefixDiskLimitMode",L"RuntimePrefixMaxAge",L"RuntimePrefixDrain",L"RuntimeDraftMode",L"RuntimeHistoryDrafts",L"RuntimeAdaptiveVerification",L"RuntimeEmbeddingStorage",L"RuntimeEmbeddingCacheMode"})
            control<ComboBox>(name).SelectionChanged([this](auto const&,auto const&){runtime_settings_changed();});
        for(auto name:{L"RuntimeVerify",L"RuntimePrefixDiskEnabled",L"RuntimeConcurrencyAuto"})
            control<ToggleSwitch>(name).Toggled([this](auto const&,auto const&){runtime_settings_changed();});
        for(auto name:{L"RuntimeMemoryValue",L"RuntimePrefixMemoryValue",L"RuntimePrefixDiskLimitValue",L"RuntimeDraftTokens",L"RuntimeEmbeddingCacheValue"})
            control<Slider>(name).ValueChanged([this](auto const&,auto const&){runtime_settings_changed();});
        for(auto name:{L"RuntimePort",L"RuntimeConcurrency"})
            control<NumberBox>(name).ValueChanged([this](auto const&,auto const&){runtime_settings_changed();});
    }
    void runtime_settings_changed() {
        if(runtime_settings_applying_)return;
        runtime_settings_dirty_=runtime_form_signature()!=runtime_settings_saved_form_;
        runtime_settings_error_.clear();runtime_settings_controls();
    }
    void runtime_settings_controls() {
        int host=control<ComboBox>(L"RuntimeHostMode").SelectedIndex();
        control<TextBox>(L"RuntimeHost").Visibility(host==2?Visibility::Visible:Visibility::Collapsed);
        text(L"RuntimeNetworkNote",host==1?L"Available to your network. Windows firewall rules apply.":host==2?L"Use an IPv4 or IPv6 address assigned to this PC.":L"Only applications on this PC can connect.");
        bool editing=runtime_settings_operation_==0;
        for(auto name:{L"RuntimeMemoryMode",L"RuntimeMemoryValue",L"RuntimeHost",L"RuntimeHostMode",L"RuntimeKvLimit",L"RuntimeConcurrencyAuto",L"RuntimeVerify",L"RuntimePort",L"RuntimePrefixDiskEnabled",L"RuntimePrefixMemoryMode",L"RuntimePrefixMaxAge",L"RuntimeDraftMode",L"RuntimeHistoryDrafts",L"RuntimeAdaptiveVerification",L"RuntimeEmbeddingStorage",L"RuntimeEmbeddingCacheMode"})control<Control>(name).IsEnabled(editing);
        bool disk=control<ToggleSwitch>(L"RuntimePrefixDiskEnabled").IsOn();
        for(auto name:{L"RuntimePrefixBrowse",L"RuntimePrefixDiskLimitMode",L"RuntimePrefixDrain"})control<Control>(name).IsEnabled(disk&&editing);
        int memory=control<ComboBox>(L"RuntimeMemoryMode").SelectedIndex();
        control<StackPanel>(L"RuntimeMemorySliderPanel").Visibility(memory?Visibility::Visible:Visibility::Collapsed);
        auto memoryValue=runtime_decimal(control<Slider>(L"RuntimeMemoryValue").Value());
        text(L"RuntimeMemoryLabel",memoryValue+(memory==3?L"% of total GPU memory":memory==2?L" GiB kept free":L" GiB occupied limit"));
        control<NumberBox>(L"RuntimeConcurrency").IsEnabled(editing&&!control<ToggleSwitch>(L"RuntimeConcurrencyAuto").IsOn());
        for(auto pair:{std::pair{L"RuntimePrefixMemoryMode",L"RuntimePrefixMemoryPanel"},std::pair{L"RuntimePrefixDiskLimitMode",L"RuntimePrefixDiskLimitPanel"},std::pair{L"RuntimeEmbeddingCacheMode",L"RuntimeEmbeddingCachePanel"}})
            control<StackPanel>(pair.second).Visibility(control<ComboBox>(pair.first).SelectedIndex()==2?Visibility::Visible:Visibility::Collapsed);
        control<StackPanel>(L"RuntimeDraftPanel").Visibility(control<ComboBox>(L"RuntimeDraftMode").SelectedIndex()==2?Visibility::Visible:Visibility::Collapsed);
        control<Slider>(L"RuntimePrefixMemoryValue").IsEnabled(editing);
        control<Slider>(L"RuntimePrefixDiskLimitValue").IsEnabled(editing&&disk);
        control<Slider>(L"RuntimeEmbeddingCacheValue").IsEnabled(editing);
        control<Slider>(L"RuntimeDraftTokens").IsEnabled(editing);
        text(L"RuntimePrefixMemoryLabel",runtime_decimal(control<Slider>(L"RuntimePrefixMemoryValue").Value())+L" GiB RAM retention");
        text(L"RuntimePrefixDiskLimitLabel",runtime_decimal(control<Slider>(L"RuntimePrefixDiskLimitValue").Value())+L" GiB on disk");
        text(L"RuntimeEmbeddingCacheLabel",runtime_decimal(control<Slider>(L"RuntimeEmbeddingCacheValue").Value())+L" GiB row cache");
        text(L"RuntimeDraftLabel",runtime_decimal(control<Slider>(L"RuntimeDraftTokens").Value())+L" draft tokens per round");
        text(L"RuntimePrefixDisk",runtime_prefix_directory_.empty()?L"Default cache directory":runtime_prefix_directory_);
        control<Button>(L"RuntimeSettingsSave").IsEnabled(runtime_settings_dirty_&&!runtime_settings_operation_&&!runtime_settings_busy_);
        control<Button>(L"RuntimeSettingsRevert").IsEnabled(runtime_settings_dirty_&&!runtime_settings_operation_);
        control<Button>(L"RuntimeSettingsReload").Visibility(runtime_reload_pending_?Visibility::Visible:Visibility::Collapsed);
        control<Button>(L"RuntimeSettingsReload").IsEnabled(!runtime_settings_dirty_&&!runtime_settings_operation_&&!runtime_settings_busy_&&session_state_==FLARE_READY);
        text(L"RuntimeSettingsMessage",!runtime_settings_error_.empty()?runtime_settings_error_:runtime_settings_operation_?L"Saving…":runtime_settings_dirty_?L"Unsaved changes. Applies on the next model load.":runtime_reload_pending_?L"Saved. Reload the model to apply runtime settings.":L"Runtime settings apply on the next model load.");
        control<TextBlock>(L"RuntimeSettingsMessage").Foreground(theme_resources_.Lookup(box_value(runtime_settings_error_.empty()?L"SecondaryTextBrush":L"AppTextBrush")).as<Media::Brush>());
    }
    void update_runtime_settings(flare_snapshot const* snapshot,flare_status const& status) {
        flare_runtime_settings value{};if(!flare_snapshot_runtime_settings(snapshot,&value))return;
        runtime_settings_busy_=status.busy!=0;runtime_reload_pending_=value.pending_reload!=0;
        if(runtime_settings_operation_&&value.last_settings_operation==runtime_settings_operation_) {
            runtime_settings_operation_=0;runtime_settings_error_=wide(value.settings_error);
            if(runtime_settings_error_.empty()){runtime_settings_dirty_=false;runtime_settings_initialized_=false;}
        }
        if(!runtime_settings_initialized_||(!runtime_settings_dirty_&&!runtime_settings_operation_)) {
            struct ApplyingReset {bool& value;~ApplyingReset(){value=false;}} applying_reset{runtime_settings_applying_};
            runtime_settings_applying_=true;
            runtime_original_={{L"memory",wide(value.memory_limit)},{L"host",wide(value.host)},{L"kv",wide(value.kv_cache_limit)},
                {L"ram",wide(value.prefix_cache_memory)},{L"directory",wide(value.prefix_cache_disk)},{L"disk_limit",wide(value.prefix_cache_disk_limit)},
                {L"age",wide(value.prefix_cache_max_age)},{L"drain",wide(value.prefix_cache_drain_on_shutdown)},
                {L"storage",wide(value.embedding_table_storage)},{L"embedding",wide(value.embedding_table_cache_size)},
                {L"concurrency",value.has_max_concurrency?std::to_wstring(value.max_concurrency):L""},
                {L"drafts",value.has_mtp_draft_tokens?std::to_wstring(value.mtp_draft_tokens):L""}};
            auto host=runtime_original_[L"host"];
            control<TextBox>(L"RuntimeHost").Text(host);
            control<ComboBox>(L"RuntimeHostMode").SelectedIndex(host==L"127.0.0.1"?0:host==L"0.0.0.0"?1:2);
            control<NumberBox>(L"RuntimePort").Value(value.port);
            auto memory=runtime_original_[L"memory"];auto first=memory.find_first_not_of(L" \t\r\n\f\v");if(first==std::wstring::npos)memory.clear();else memory.erase(0,first);
            int mode=memory.empty()?0:memory.front()==L'-'?2:memory.find_first_of(L"KkMmGgTtBb")!=std::wstring::npos?1:3;
            double displayed=mode==3?std::stod(memory)*100:mode?runtime_size_gib(memory):2;
            if(mode==3)runtime_memory_percent_=displayed;else runtime_memory_gib_=displayed;
            control<ComboBox>(L"RuntimeMemoryMode").SelectedIndex(mode);runtime_memory_slider(mode,displayed);
            auto kv=runtime_original_[L"kv"];control<ComboBox>(L"RuntimeKvLimit").SelectedIndex(kv==L"context"?1:kv==L"concurrency"?2:0);
            control<ToggleSwitch>(L"RuntimeConcurrencyAuto").IsOn(!value.has_max_concurrency);
            control<NumberBox>(L"RuntimeConcurrency").Maximum(std::max(64.0,static_cast<double>(value.max_concurrency)));
            control<NumberBox>(L"RuntimeConcurrency").Value(value.has_max_concurrency?static_cast<double>(value.max_concurrency):1);
            text(L"RuntimeConcurrencyNote",L"Automatic uses the model and configuration defaults. A manual limit must fit the prepared model.");
            flare_model_detail detail{};
            if(flare_snapshot_model_detail(snapshot,&detail)&&(detail.known_fields&FLARE_DETAIL_CONCURRENCY)) {
                bool active=false;
                for(size_t i=0;i<flare_snapshot_model_count(snapshot);++i){flare_model model{};if(flare_snapshot_model(snapshot,i,&model)&&(model.flags&FLARE_MODEL_ACTIVE)&&utf8(model.id)==utf8(detail.id)){active=true;break;}}
                if(active&&detail.max_concurrency){control<NumberBox>(L"RuntimeConcurrency").Maximum(std::max(static_cast<double>(detail.max_concurrency),value.has_max_concurrency?static_cast<double>(value.max_concurrency):1.0));text(L"RuntimeConcurrencyNote",L"Prepared model maximum: "+std::to_wstring(detail.max_concurrency)+L" requests. Automatic keeps configuration defaults.");}
            }
            control<ToggleSwitch>(L"RuntimeVerify").IsOn(value.verify_weights!=0);
            control<ToggleSwitch>(L"RuntimePrefixDiskEnabled").IsOn(value.prefix_cache_disk_enabled!=0);
            runtime_prefix_directory_=runtime_original_[L"directory"];
            auto budget=[&](wchar_t const* key,wchar_t const* choice,wchar_t const* slider,double maximum,double fallback){auto raw=runtime_original_[key];double amount=runtime_size_gib(raw);control<ComboBox>(choice).SelectedIndex(raw.empty()?0:amount==0?1:2);runtime_slider(slider,amount>0?amount:fallback,0.125,maximum,0.125);};
            budget(L"ram",L"RuntimePrefixMemoryMode",L"RuntimePrefixMemoryValue",64,1);
            budget(L"disk_limit",L"RuntimePrefixDiskLimitMode",L"RuntimePrefixDiskLimitValue",1024,16);
            budget(L"embedding",L"RuntimeEmbeddingCacheMode",L"RuntimeEmbeddingCacheValue",64,1);
            runtime_duration(L"RuntimePrefixMaxAge",runtime_original_[L"age"],{L"off",L"1h",L"24h",L"168h"});
            runtime_duration(L"RuntimePrefixDrain",runtime_original_[L"drain"],{L"off",L"30s",L"2m",L"on"});
            auto storage=runtime_original_[L"storage"];control<ComboBox>(L"RuntimeEmbeddingStorage").SelectedIndex(storage==L"disk"?1:storage==L"ram"?2:0);
            control<ComboBox>(L"RuntimeDraftMode").SelectedIndex(!value.has_mtp_draft_tokens?0:value.mtp_draft_tokens==0?1:2);
            runtime_slider(L"RuntimeDraftTokens",value.has_mtp_draft_tokens&&value.mtp_draft_tokens?static_cast<double>(value.mtp_draft_tokens):1,1,64,1);
            control<ComboBox>(L"RuntimeHistoryDrafts").SelectedIndex(value.history_drafts);
            control<ComboBox>(L"RuntimeAdaptiveVerification").SelectedIndex(value.adaptive_verification);
            runtime_field_baseline_.clear();
            for(auto key:{L"memory",L"host",L"kv",L"concurrency",L"ram",L"directory",L"disk_limit",L"age",L"drain",L"storage",L"embedding",L"drafts"})runtime_field_baseline_[key]=runtime_field_signature(key);
            runtime_settings_saved_form_=runtime_form_signature();runtime_settings_initialized_=true;
        }
        text(L"RuntimeAuthentication",value.has_api_key?L"API key configured in Flare configuration.":L"No API key configured.");
        auto bind=wide(value.active_bind);text(L"RuntimeActiveBind",bind.empty()?L"Server stopped":L"Listening: "+bind);
        runtime_settings_controls();
    }
    void save_runtime_settings() {
        if(runtime_settings_operation_||runtime_settings_busy_)return;
        try {
            std::vector<std::string> strings;strings.reserve(16);
            auto store=[&](std::wstring raw){strings.push_back(to_string(hstring(raw)));auto const& s=strings.back();return flare_string{s.data(),s.size()};};
            flare_runtime_settings value{};
            int memory=control<ComboBox>(L"RuntimeMemoryMode").SelectedIndex();
            double amount=control<Slider>(L"RuntimeMemoryValue").Value();
            auto memoryText=memory==0?L"":memory==3?runtime_decimal(amount/100):std::wstring(memory==2?L"-":L"")+runtime_decimal(amount)+L"GiB";
            value.memory_limit=store(runtime_preserve(L"memory",memoryText));
            int kv=control<ComboBox>(L"RuntimeKvLimit").SelectedIndex();
            value.kv_cache_limit=store(runtime_preserve(L"kv",kv==1?L"context":kv==2?L"concurrency":L"memory"));
            int host=control<ComboBox>(L"RuntimeHostMode").SelectedIndex();
            value.host=store(runtime_preserve(L"host",host==2?std::wstring(control<TextBox>(L"RuntimeHost").Text()):host==1?L"0.0.0.0":L"127.0.0.1"));
            auto port=control<NumberBox>(L"RuntimePort").Value();
            if(!std::isfinite(port)||port!=std::floor(port)||port<1||port>65535)throw std::invalid_argument("Port must be a whole number from 1 to 65535.");
            value.port=static_cast<uint32_t>(port);value.verify_weights=control<ToggleSwitch>(L"RuntimeVerify").IsOn();
            double concurrency=control<NumberBox>(L"RuntimeConcurrency").Value();
            bool automatic=control<ToggleSwitch>(L"RuntimeConcurrencyAuto").IsOn();
            if(!automatic&&(!std::isfinite(concurrency)||concurrency!=std::floor(concurrency)||concurrency<1||concurrency>64))throw std::invalid_argument("Concurrency must be a whole number from 1 to 64.");
            auto concurrencyText=runtime_preserve(L"concurrency",automatic?L"":runtime_decimal(concurrency));
            value.has_max_concurrency=!concurrencyText.empty();if(value.has_max_concurrency)value.max_concurrency=std::stoull(concurrencyText);
            int drafts=control<ComboBox>(L"RuntimeDraftMode").SelectedIndex();
            auto draftText=runtime_preserve(L"drafts",drafts==0?L"":drafts==1?L"0":runtime_decimal(std::round(control<Slider>(L"RuntimeDraftTokens").Value())));
            value.has_mtp_draft_tokens=!draftText.empty();if(value.has_mtp_draft_tokens)value.mtp_draft_tokens=std::stoull(draftText);
            value.prefix_cache_disk_enabled=control<ToggleSwitch>(L"RuntimePrefixDiskEnabled").IsOn();
            value.prefix_cache_memory=store(runtime_preserve(L"ram",runtime_budget_value(L"RuntimePrefixMemoryMode",L"RuntimePrefixMemoryValue")));
            value.prefix_cache_disk=store(runtime_preserve(L"directory",runtime_prefix_directory_));
            value.prefix_cache_disk_limit=store(runtime_preserve(L"disk_limit",runtime_budget_value(L"RuntimePrefixDiskLimitMode",L"RuntimePrefixDiskLimitValue")));
            int age=control<ComboBox>(L"RuntimePrefixMaxAge").SelectedIndex();
            std::vector<std::wstring> ages{L"off",L"1h",L"24h",L"168h"};
            value.prefix_cache_max_age=store(runtime_preserve(L"age",age>=0&&age<4?ages[age]:runtime_original_[L"age"]));
            int drain=control<ComboBox>(L"RuntimePrefixDrain").SelectedIndex();
            std::vector<std::wstring> drains{L"off",L"30s",L"2m",L"on"};
            value.prefix_cache_drain_on_shutdown=store(runtime_preserve(L"drain",drain>=0&&drain<4?drains[drain]:runtime_original_[L"drain"]));
            int storage=control<ComboBox>(L"RuntimeEmbeddingStorage").SelectedIndex();
            value.embedding_table_storage=store(runtime_preserve(L"storage",storage==1?L"disk":storage==2?L"ram":L"auto"));
            value.embedding_table_cache_size=store(runtime_preserve(L"embedding",runtime_budget_value(L"RuntimeEmbeddingCacheMode",L"RuntimeEmbeddingCacheValue")));
            value.history_drafts=control<ComboBox>(L"RuntimeHistoryDrafts").SelectedIndex();
            value.adaptive_verification=control<ComboBox>(L"RuntimeAdaptiveVerification").SelectedIndex();
            runtime_settings_operation_=flare_app_set_runtime_settings(app_,&value);
            runtime_settings_error_=runtime_settings_operation_?L"":wide(flare_last_error());
        }catch(std::exception const& error){runtime_settings_error_=std::wstring(to_hstring(error.what()));}
        catch(hresult_error const& error){runtime_settings_error_=std::wstring(error.message());}
        runtime_settings_controls();
    }
    void choose_prefix_directory() {
        try {
            com_ptr<IFileOpenDialog> picker;check_hresult(CoCreateInstance(CLSID_FileOpenDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(picker.put())));
            DWORD options{};check_hresult(picker->GetOptions(&options));check_hresult(picker->SetOptions(options|FOS_PICKFOLDERS|FOS_FORCEFILESYSTEM|FOS_PATHMUSTEXIST));
            check_hresult(picker->SetTitle(L"Choose prefix cache directory"));
            auto result=picker->Show(hwnd_);if(result==HRESULT_FROM_WIN32(ERROR_CANCELLED))return;check_hresult(result);
            com_ptr<IShellItem> item;check_hresult(picker->GetResult(item.put()));wchar_t* path{};check_hresult(item->GetDisplayName(SIGDN_FILESYSPATH,&path));
            runtime_prefix_directory_=path;CoTaskMemFree(path);runtime_settings_changed();
        }catch(hresult_error const& error){show_error(std::wstring(error.message()));}
    }
